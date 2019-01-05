from collections import OrderedDict
from shared_optim import SharedRMSprop, SharedAdam
import torch.optim as optim
from utils import *
from ggm import ggm
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import time
import random
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import pickle
import argparse

def train(shared_model, optimizer, smiles, scaffold, condition1, condition2, pid, retval_list):

    #optimizer = optim.Adam(shared_model.parameters(), lr=1e-4)
    model=ggm()
    st1 = time.time()
    for idx in range(len(smiles)):
        st = time.time()
        model.load_state_dict(shared_model.state_dict())
        model.zero_grad()
        optimizer.zero_grad()
        #print (id_to_smiles[k][0], id_to_smiles[k][1])
        #retval = model('[N-]=[N+]=C1/C(=N/N=C/c2ccc(cc2)Cl)/N(C)C(=O)N(C1=O)C', '[N+]=C1C(=O)NC(=O)NC1=NN=Cc1ccccc1', id_to_activity[k])
        retval = model(smiles[idx], scaffold[idx], condition1[idx], condition2[idx])
        #retval = shared_model(s)
        if retval is None:
            continue
        g_gen, h_gen, loss1, loss2, loss3 = retval
        loss = loss1 + loss2 + loss3
        retval_list[pid].append((loss.data.cpu().numpy()[0], loss1.data.cpu().numpy()[0], loss2.data.cpu().numpy()[0], loss3.data.cpu().numpy()[0]))
        #print (idx, loss1.data.cpu().numpy()[0], loss2.data.cpu().numpy()[0], loss3.data.cpu().numpy()[0,0])
        #optimizer.zero_grad()
        end = time.time()
        st = time.time()
        loss.backward()
        end = time.time()

        #torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        ensure_shared_grads(model, shared_model, True)
        #for p in model.parameters():
        #    if p.grad is None:
        #        print ('None')

        optimizer.step()

        end = time.time()
        #if pid==0:
        #    print (idx, loss.data.cpu().numpy()[0])
    end1 = time.time()

    #print ('accumulate time', pid, end1-st1)



if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--lr', help="learning rate", type=float, default = 1e-4)
    parser.add_argument('--num_epochs', help='number of epochs', type = int, default = 10000)
    parser.add_argument('--ncpus', help = 'number of cpus', type = int) 
    parser.add_argument('--item_per_cycle', help = 'iteration per cycle', type = int, default = 128) 
    args = parser.parse_args()
    
    #hyperparameters
    num_epochs = args.num_epochs
    ncpus = args.ncpus
    item_per_cycle = args.item_per_cycle
    lr = args.lr
     
    #lines for multiprocessing
    mp.set_start_method('spawn')
    torch.manual_seed(1)

    #model 
    shared_model = ggm()
    shared_model.share_memory()


    #shared optimizer
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=lr, amsgrad=True)
    shared_optimizer.share_memory()
    print ("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
    
    #initialize parameters of the model 
    shared_model = initialize_model(shared_model, False)


    #load data and keys
    with open('./egfr_keys/train_active_keys.pkl', 'rb') as f:
        train_active_keys = pickle.load(f)
    with open('./egfr_keys/train_inactive_keys.pkl', 'rb') as f:
        train_inactive_keys = pickle.load(f)
    with open('./egfr_keys/test_active_keys.pkl', 'rb') as f:
        test_active_keys = pickle.load(f)
    with open('./egfr_keys/test_inactive_keys.pkl', 'rb') as f:
        test_inactive_keys = pickle.load(f)
    with open('./egfr_keys/id_to_smiles.pkl', 'rb') as f:
        id_to_smiles = pickle.load(f)
    with open('./egfr_keys/id_to_whole_activity.pkl', 'rb') as f:
        id_to_whole_activity = pickle.load(f)
    with open('./egfr_keys/id_to_scaffold_activity.pkl', 'rb') as f:
        id_to_scaffold_activity = pickle.load(f)
    

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
                condition1 = [id_to_whole_activity[k] for k in keys]
                condition2 = [id_to_scaffold_activity[k] for k in keys]

                #we need smiles of whole molecule and scaffold to make graph of a molecule                
                smiles = [id_to_smiles[k][0] for k in keys]
                scaffold = [id_to_smiles[k][1] for k in keys]

                p = mp.Process(target=train, args=(shared_model, shared_optimizer, smiles, scaffold, condition1, condition2, i, retval))
                p.start()
                processes.append(p)
                time.sleep(0.1)
            for p in processes:
                p.join() 
            end = time.time()       
            for k in retval:
                #print (len(k))
                for j in k:
                    #j[0] : total loss
                    #j[1] : add node, select node, edge type loss
                    #j[2] : vae loss
                    #j[3] : isomer loss
                    print (count, epoch, c, j[0], j[1], j[2], j[3], end-st)
                    count+=1
            name = 'save/save_check_'+str(epoch)+'_' + str(c)+'.pt'
            torch.save(shared_model.state_dict(), name)




