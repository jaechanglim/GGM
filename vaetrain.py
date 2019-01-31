import argparse
from collections import OrderedDict
import os
import random
import re
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import torch.nn as nn
import torch.optim as optim

from ggm import ggm
from shared_optim import SharedRMSprop, SharedAdam
import utils

def train(shared_model, optimizer, wholes, scaffolds, whole_conditions, scaffold_conditions, pid, retval_list, args):
    """\
    Target function for the multiprocessed training.

    In addition to updating model parameters, 
    loss values are collected by `retval_list` after each `forward`.

    Parameters
    ----------
    shared_model: torch.nn.Module
        A shared model to be trained.
    optimizer: torch.optim.Optimizer
        A shared optimizer.
    wholes: list[str]
        A list of whole-molecule SMILESs.
    scaffolds: list[str]
        A list of scaffold SMILESs.
    whole_conditions: list[ list[float] ]
        [
            [ value1, value2, ... ],  # condition values of whole 1
            [ value1, value2, ... ],  # condition values of whole 2
        ]
    scaffold_conditions: list[ list[float] ]
        Similar to `whole_conditions`, but with scaffold values.
    pid: int
        CPU index.
    retval_list: list[multiprocessing.managers.ListProxy]
        A list of lists to collect loss floats from CPUs.
        In each cycle, the final shape will be:
            (ncpus, minibatch_size, num_of_losses)
    args: argparse.Namespace
        Delivers hyperparameters from command arguments to the model.
    """
    #each thread make new model
    # Number of conditions <- (number of properties) * 2
    model=ggm(args, len(args.key_dirs)*2)
    for idx in range(len(wholes)):
        #set parameters of model as same as that of reference model
        model.load_state_dict(shared_model.state_dict())
        model.zero_grad()
        optimizer.zero_grad()
        
        #forward
        retval = model(wholes[idx], scaffolds[idx], whole_conditions[idx], scaffold_conditions[idx], args.shuffle_order)
        
        #if retval is None, some error occured. it is usually due to invalid smiles
        if retval is None:
            continue

        #train model
        g_gen, h_gen, loss1, loss2, loss3 = retval
        loss = loss1 + loss2*args.beta1 + loss3  # torch.autograd.Variable of shape (1,)
        retval_list[pid].append((loss.data.cpu().numpy()[0], loss1.data.cpu().numpy()[0], loss2.data.cpu().numpy()[0], loss3.data.cpu().numpy()[0]))
        loss.backward()

        #torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        utils.ensure_shared_grads(model, shared_model, True)
        optimizer.step()

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--lr', help="learning rate", type=float, default = 1e-4)
    parser.add_argument('--num_epochs', help='number of epochs', type = int, default = 10000)
    parser.add_argument('--ncpus', help = 'number of cpus', type = int, default = 1) 
    parser.add_argument('--item_per_cycle', help = 'iteration per cycle', type = int, default = 128) 
    parser.add_argument('--dim_of_node_vector', help = 'dimension of node_vector', type = int, default = 128) 
    parser.add_argument('--dim_of_edge_vector', help = 'dimension of edge vector', type = int, default = 128) 
    parser.add_argument('--dim_of_FC', help = 'dimension of FC', type = int, default = 128) 
    parser.add_argument('--save_every', help = 'choose how often model will be saved', type = int, default = 200) 
    parser.add_argument('--beta1', help = 'beta1: lambda paramter for VAE training', type = float, default = 5e-3) 
    parser.add_argument('--save_dir', help = 'save directory', type = str) 
    parser.add_argument('--key_dirs', help='key directories', nargs='+', metavar='PATH')
    parser.add_argument('--shuffle_order', help = 'shuffle order or adding node and edge', action='store_true') 
    parser.add_argument('--active_ratio', help='active ratio in sampling (default: no matter)', type=float)
    parser.add_argument('--save_fpath', help='file path of a saved model to restart') 
    args = parser.parse_args()
    args.save_dir = os.path.expanduser(args.save_dir)
    key_dirs = [os.path.expanduser(path) for path in args.key_dirs]
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    #hyperparameters
    num_epochs = args.num_epochs
    ncpus = args.ncpus
    item_per_cycle = args.item_per_cycle  # Num of data per cycle per CPU.
    lr = args.lr
    save_every=args.save_every
     
    #lines for multiprocessing
    mp.set_start_method('spawn')
    torch.manual_seed(1)

    #model 
    # Number of conditions <- (number of properties) * 2
    shared_model = ggm(args, len(args.key_dirs)*2)
    shared_model.share_memory()  # torch.nn.Module.share_memory

    #shared optimizer
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=lr, amsgrad=True)
    shared_optimizer.share_memory()
    print ("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
    
    #initialize parameters of the model 
    if args.save_fpath:
        args.save_fpath = os.path.expanduser(args.save_fpath)
        initial_epoch, initial_cycle = [int(value) for value in re.findall('\d+', args.save_fpath)]
        shared_model = utils.initialize_model(shared_model, args.save_fpath)
    else:
        initial_epoch = initial_cycle = 0
        shared_model = utils.initialize_model(shared_model, False)

    # Load data and keys.
    # See `utils.load_data` for the variable structures.
    ( id_to_smiles,
      id_to_whole_conditions, id_to_scaffold_conditions,
      active_keys, inactive_keys ) = utils.load_data(key_dirs, 'train')

    num_cycles = int(len(id_to_smiles)/ncpus/item_per_cycle)
    print(f"""\
ncpus             : {ncpus}
OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS')}
Number of data    : {len(id_to_smiles)}
Number of epochs  : {num_epochs}
Number of cycles  : {num_cycles} per epoch
Minibatch size    : {item_per_cycle} per CPU per cycle
Save model every  : {save_every} cycles per epoch (Total {num_epochs*(num_cycles//save_every+1)} models)
Save directory    : {os.path.abspath(args.save_dir)}
beta1             : {args.beta1}
Learning rate     : {lr}
dim_of_node_vector: {args.dim_of_node_vector}
dim_of_edge_vector: {args.dim_of_edge_vector}
dim_of_FC         : {args.dim_of_FC}
shuffle_order     : {args.shuffle_order}
Data directories  : {key_dirs}
Restart from      : {os.path.abspath(args.save_fpath) if args.save_fpath else None}
""")
    
    print("epoch  cyc  totcyc  loss  loss1  loss2  loss3  time")
    for epoch in range(num_epochs):
        # Jump to the previous epoch for restart.
        if epoch < initial_epoch:
            continue
        for cycle in range(num_cycles):
            # Jump to the previous cycle for restart.
            if epoch == initial_epoch:
                if cycle < initial_cycle:
                    continue
            retval_list = mp.Manager().list()  # Is this needed?
            # List of multiprocessing.managers.ListProxy to collect losses
            retval_list = [mp.Manager().list() for i in range(ncpus)]
            st = time.time()
            processes = []
            for pid in range(ncpus):
                # Sample keys without considering activeness.
                if args.active_ratio is None:
                    keys = random.sample(id_to_smiles.keys(), item_per_cycle)
                # Sample active and inactive keys by the required ratio.
                else:
                    keys = utils.sample_data(acaive_keys, inactive_keys, item_per_cycle, args.active_ratio)

                # Property (descriptor) values work as conditions;
                # we need both values of whole molecules and scaffolds.
                # whole_conditions := [
                #     [ value1, value2, ... ],  # condition values of whole 1
                #     [ value1, value2, ... ],  # condition values of whole 2
                #     ... ]
                whole_conditions = [id_to_whole_conditions[key] for key in keys]
                scaffold_conditions = [id_to_scaffold_conditions[key] for key in keys]

                # We need SMILESs of whole molecules and scaffolds to make molecule graphs.
                wholes = [id_to_smiles[key][0] for key in keys]     # list of whole SMILESs
                scaffolds = [id_to_smiles[key][1] for key in keys]  # list of scaffold SMILESs

                proc = mp.Process(target=train, args=(shared_model, shared_optimizer, wholes, scaffolds, whole_conditions, scaffold_conditions, pid, retval_list, args))
                proc.start()
                processes.append(proc)
                time.sleep(0.1)
            for proc in processes:
                proc.join() 
            end = time.time()

            # retval_list shape -> (ncpus, item_per_cycle, 4),
            loss = np.mean(np.array([losses[0] for k in retval_list for losses in k]))
            loss1 = np.mean(np.array([losses[1] for k in retval_list for losses in k]))
            loss2 = np.mean(np.array([losses[2] for k in retval_list for losses in k]))
            loss3 = np.mean(np.array([losses[3] for k in retval_list for losses in k]))
            print ('%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' %(epoch, cycle, epoch*num_cycles+cycle, loss, loss1, loss2, loss3, end-st))
            if cycle%save_every == 0:
                name = args.save_dir+'/save_'+str(epoch)+'_' + str(cycle)+'.pt'
                torch.save(shared_model.state_dict(), name)
