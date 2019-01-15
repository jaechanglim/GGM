import argparse
from collections import OrderedDict
import os
import pickle
import random
from random import shuffle
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import torch.nn as nn
import torch.optim as optim

from ggm import ggm
from shared_optim import SharedRMSprop, SharedAdam
from utils import *

def train(shared_model, optimizer, smiles, scaffold, condition1, condition2, pid, retval_list, args):
    #each thread make new model
    model=ggm(args)
    for idx in range(len(smiles)):
        #set parameters of model as same as that of reference model
        model.load_state_dict(shared_model.state_dict())
        model.zero_grad()
        optimizer.zero_grad()
        
        #forward
        retval = model(smiles[idx], scaffold[idx], condition1[idx], condition2[idx], args.beta1)
        
        #if retval is None, some error occured. it is usually due to invalid smiles
        if retval is None:
            continue

        #train model
        g_gen, h_gen, loss1, loss2, loss3 = retval
        loss = loss1 + loss2 + loss3
        retval_list[pid].append((loss.data.cpu().numpy()[0], loss1.data.cpu().numpy()[0], loss2.data.cpu().numpy()[0], loss3.data.cpu().numpy()[0]))
        loss.backward()

        #torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        ensure_shared_grads(model, shared_model, True)
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
    parser.add_argument('--beta1', help = 'beta1 : lambda paramter for VAE training', type = int, default = 5e-3) 
    parser.add_argument('--save_dir', help = 'save directory', type = str) 
    parser.add_argument('--key_dir', help = 'key directory', type = str) 
    args = parser.parse_args()
    if not os.path.isdir(args.save_dir):
        os.system('mkdir ' + args.save_dir)
    #hyperparameters
    num_epochs = args.num_epochs
    ncpus = args.ncpus
    item_per_cycle = args.item_per_cycle
    lr = args.lr
    save_every=args.save_every
     
    #lines for multiprocessing
    mp.set_start_method('spawn')
    torch.manual_seed(1)

    #model 
    shared_model = ggm(args)
    shared_model.share_memory()


    #shared optimizer
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=lr, amsgrad=True)
    shared_optimizer.share_memory()
    print ("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
    
    #initialize parameters of the model 
    shared_model = initialize_model(shared_model, False)


    #load data and keys
    with open(args.key_dir+'/train_active_keys.pkl', 'rb') as f:
        train_active_keys = pickle.load(f)
    with open(args.key_dir+'/train_inactive_keys.pkl', 'rb') as f:
        train_inactive_keys = pickle.load(f)
    with open(args.key_dir+'/test_active_keys.pkl', 'rb') as f:
        test_active_keys = pickle.load(f)
    with open(args.key_dir+'/test_inactive_keys.pkl', 'rb') as f:
        test_inactive_keys = pickle.load(f)
    with open(args.key_dir+'/id_to_smiles.pkl', 'rb') as f:
        id_to_smiles = pickle.load(f)
    with open(args.key_dir+'/id_to_condition1.pkl', 'rb') as f:
        id_to_condition1 = pickle.load(f)
    with open(args.key_dir+'/id_to_condition2.pkl', 'rb') as f:
        id_to_condition2 = pickle.load(f)
    

    num_cycles = int(len(id_to_smiles)/ncpus/item_per_cycle)
    
    count = 0
    for epoch in range(num_epochs):
        for c in range(num_cycles):
            retval = mp.Manager().list()
            retval = [mp.Manager().list() for i in range(ncpus)]
            st = time.time()
            processes = []
            for i in range(ncpus):
                #sample active and inactive keys with same ratio
                keys1 = [random.choice(train_active_keys) for i in range(int(item_per_cycle/2))]
                keys2 = [random.choice(train_inactive_keys) for i in range(int(item_per_cycle/2))]
                keys = keys1 + keys2
                random.shuffle(keys)

                #activities work as condition. we need both activities of whole molecule and scaffold
                condition1 = [id_to_condition1[k] for k in keys]
                condition2 = [id_to_condition2[k] for k in keys]

                #we need smiles of whole molecule and scaffold to make graph of a molecule                
                smiles = [id_to_smiles[k][0] for k in keys]
                scaffold = [id_to_smiles[k][1] for k in keys]

                p = mp.Process(target=train, args=(shared_model, shared_optimizer, smiles, scaffold, condition1, condition2, i, retval, args))
                p.start()
                processes.append(p)
                time.sleep(0.1)
            for p in processes:
                p.join() 
            end = time.time()       
            loss = np.mean(np.array([j[0] for k in retval for j in k]))
            loss1 = np.mean(np.array([j[1] for k in retval for j in k]))
            loss2 = np.mean(np.array([j[2] for k in retval for j in k]))
            loss3 = np.mean(np.array([j[3] for k in retval for j in k]))
            print ('%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' %(epoch, c, epoch*num_cycles+c, loss, loss1, loss2, loss3, end-st))
            if c%save_every==0:
                name = args.save_dir+'/save_'+str(epoch)+'_' + str(c)+'.pt'
                torch.save(shared_model.state_dict(), name)




