import argparse
from collections import OrderedDict
import os
import pickle
import random
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

def train(shared_model, optimizer, smiles, scaffold, condition1, condition2, pid, retval_list, args):
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
    smiles: list[str]
        A list of whole SMILESs.
    scaffold: list[str]
        A list of scaffold SMILESs corresponding to `smiles`.
    condition1: list[float]
        A list of whole property values for CVAE.
    condition2: list[float]
        A list of scaffold property values for CVAE.
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
    model=ggm(args)
    for idx in range(len(smiles)):
        #set parameters of model as same as that of reference model
        model.load_state_dict(shared_model.state_dict())
        model.zero_grad()
        optimizer.zero_grad()
        
        #forward
        retval = model(smiles[idx], scaffold[idx], condition1[idx], condition2[idx], args.shuffle_order)
        
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
    parser.add_argument('--key_dir', help = 'key directory', type = str) 
    parser.add_argument('--shuffle_order', help = 'shuffle order or adding node and edge', dest='shuffle_order', action='store_true') 
    args = parser.parse_args()
    args.save_dir = os.path.expanduser(args.save_dir)
    args.key_dir = os.path.expanduser(args.key_dir)
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
    shared_model = ggm(args)
    shared_model.share_memory()  # torch.nn.Module.share_memory

    #shared optimizer
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=lr, amsgrad=True)
    shared_optimizer.share_memory()
    print ("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
    
    #initialize parameters of the model 
    shared_model = utils.initialize_model(shared_model, False)

    #load data and keys
    with open(args.key_dir+'/train_active_keys.pkl', 'rb') as f:
        train_active_keys = pickle.load(f)
        # list[str] -> ['STOCK4S-00033', 'STOCK4S-00102', ...]
    with open(args.key_dir+'/train_inactive_keys.pkl', 'rb') as f:
        train_inactive_keys = pickle.load(f)
        # list[str] -> ['STOCK4S-00033', 'STOCK4S-00102', ...]
    with open(args.key_dir+'/test_active_keys.pkl', 'rb') as f:
        test_active_keys = pickle.load(f)
        # list[str] -> ['STOCK4S-00033', 'STOCK4S-00102', ...]
    with open(args.key_dir+'/test_inactive_keys.pkl', 'rb') as f:
        test_inactive_keys = pickle.load(f)
        # list[str] -> ['STOCK4S-00033', 'STOCK4S-00102', ...]
    with open(args.key_dir+'/id_to_smiles.pkl', 'rb') as f:
        id_to_smiles = pickle.load(f)
        # dict[str, list[str]] -> {'STOCK4S-67369': [whole_SMILES, scaffold_SMILES], ...}
    with open(args.key_dir+'/id_to_condition1.pkl', 'rb') as f:
        id_to_condition1 = pickle.load(f)
        # dict[str, list[float]] -> {'STOCK4S-67369': [whole_value, 1-whole_value], ...}
    with open(args.key_dir+'/id_to_condition2.pkl', 'rb') as f:
        id_to_condition2 = pickle.load(f)
        # dict[str, list[float]] -> {'STOCK4S-67369': [scaffold_value, 1-scaffold_value], ...}
        # The property values (affinity, TPSA, logP and MW) are all normalized within [0, 1].

    num_cycles = int(len(id_to_smiles)/ncpus/item_per_cycle)
    print(f"""\
ncpus             : {ncpus}
OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS')}
Number of data    : {len(id_to_smiles)}
Number of epochs  : {num_epochs}
Number of cycles  : {num_cycles} per epoch
Minibatch size    : {item_per_cycle} per CPU per cycle
Save model every  : {save_every} cycles per epoch (Total {num_epochs*(num_cycles//save_every+1)} models)
beta1             : {args.beta1}
Learning rate     : {lr}
dim_of_node_vector: {args.dim_of_node_vector}
dim_of_edge_vector: {args.dim_of_edge_vector}
dim_of_FC         : {args.dim_of_FC}
shuffle_order     : {args.shuffle_order}
""")
    
    print("# epoch  cycle_in_epoch  total_cycle  loss  loss1  loss2  loss3  time")
    for epoch in range(num_epochs):
        for c in range(num_cycles):
            retval_list = mp.Manager().list()  # Is this needed?
            # List of multiprocessing.managers.ListProxy to collect losses
            retval_list = [mp.Manager().list() for i in range(ncpus)]
            st = time.time()
            processes = []
            for i in range(ncpus):
                #sample active and inactive keys with same ratio
                keys1 = [random.choice(train_active_keys) for i in range(int(item_per_cycle/2))]
                keys2 = [random.choice(train_inactive_keys) for i in range(int(item_per_cycle/2))]
                keys = keys1 + keys2
                random.shuffle(keys)

                #activities work as condition. we need both activities of whole molecule and scaffold
                condition1 = [id_to_condition1[k] for k in keys]  # [[whole_value, 1-whole_value], ...]
                condition2 = [id_to_condition2[k] for k in keys]  # [[scaffold_value, 1-scaffold_value], ...]

                #we need smiles of whole molecule and scaffold to make graph of a molecule                
                smiles = [id_to_smiles[k][0] for k in keys]    # list of whole SMILESs
                scaffold = [id_to_smiles[k][1] for k in keys]  # list of scaffold SMILESs

                p = mp.Process(target=train, args=(shared_model, shared_optimizer, smiles, scaffold, condition1, condition2, i, retval_list, args))
                p.start()
                processes.append(p)
                time.sleep(0.1)
            for p in processes:
                p.join() 
            end = time.time()

            # retval_list shape -> (ncpus, item_per_cycle, 4),
            loss = np.mean(np.array([losses[0] for k in retval_list for losses in k]))
            loss1 = np.mean(np.array([losses[1] for k in retval_list for losses in k]))
            loss2 = np.mean(np.array([losses[2] for k in retval_list for losses in k]))
            loss3 = np.mean(np.array([losses[3] for k in retval_list for losses in k]))
            print ('%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' %(epoch, c, epoch*num_cycles+c, loss, loss1, loss2, loss3, end-st))
            if c%save_every==0:
                name = args.save_dir+'/save_'+str(epoch)+'_' + str(c)+'.pt'
                torch.save(shared_model.state_dict(), name)
