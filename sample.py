import argparse
from collections import OrderedDict
from random import shuffle
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import torch.nn as nn
import torch.optim as optim

from ggm import ggm
from shared_optim import SharedRMSprop, SharedAdam
import utils

def normalize(v, max_v, min_v):
    v = min(max_v, v)
    v = max(min_v, v)
    return (v-min_v)/(max_v-min_v)

#c1ccc(Nc2ncnc3c2oc2ccccc23)cc1 CHEMBL149512 Brc4cccc(Nc2ncnc3c1ccccc1oc23)c4 6.130768280269024

def sample(shared_model, smiles, scaffold, condition1, condition2, pid, retval_list, args):
    #optimizer = optim.Adam(shared_model.parameters(), lr=1e-4)
    model=ggm(args)
    st1 = time.time()

    for idx, (s1,s2) in enumerate(zip(smiles, scaffold)):
        model.load_state_dict(shared_model.state_dict())
        retval = model.sample(None, s2, latent_vector=None, condition1 = condition1, condition2 = condition2, stochastic=False)
        #retval = model.sample(s1, s2, latent_vector=None, stochastic=False)
        #retval = shared_model(s)
        if retval is None:
            continue
        g_gen, h_gen  = retval
        
        try:
            new_s = utils.graph_to_smiles(g_gen, h_gen)
        except:
            new_s = None
        retval_list[pid].append((s1, new_s))
    end1 = time.time()

    #print ('accumulate time', pid, end1-st1)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--ncpus', help = 'number of cpus', type = int, default = 1) 
    parser.add_argument('--item_per_cycle', help = 'iteration per cycle', type = int, default = 128) 
    parser.add_argument('--dim_of_node_vector', help = 'dimension of node_vector', type = int, default = 128) 
    parser.add_argument('--dim_of_edge_vector', help = 'dimension of edge vector', type = int, default = 128) 
    parser.add_argument('--dim_of_FC', help = 'dimension of FC', type = int, default = 128) 
    parser.add_argument('--save_fpath', help = 'file path of saved model', type = str) 
    parser.add_argument('--scaffold', help = 'smiles of scaffold', type = str) 
    parser.add_argument('--target_property', help = 'value of target property', type = float) 
    parser.add_argument('--scaffold_property', help = 'valud of scaffold property', type = float) 
    parser.add_argument('--output_filename', help = 'output file name', type = str) 
    parser.add_argument('--minimum_value', help = 'minimum value of property. It will be used for nomralization', type = float) 
    parser.add_argument('--maximum_value', help = 'maximum value of property. It will be used for nomralization', type = float) 
    args = parser.parse_args()
    
    #hyperparameters
    ncpus = args.ncpus
    item_per_cycle = args.item_per_cycle
    save_fpath = args.save_fpath
    target_property=normalize(args.target_property, args.maximum_value, args.minimum_value)
    scaffold_property=normalize(args.scaffold_property, args.maximum_value, args.minimum_value)

    #lines for multiprocessing
    mp.set_start_method('spawn')
    torch.manual_seed(1)

    #model 
    shared_model = ggm(args)
    shared_model.share_memory()

    print ("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
    
    #initialize parameters of the model 
    shared_model = utils.initialize_model(shared_model, save_fpath)
    
    scaffold = args.scaffold
    scaffold = [scaffold for i in range(item_per_cycle)]
    smiles = [None for i in range(item_per_cycle)]
    retval = mp.Manager().list()
    retval = [mp.Manager().list() for i in range(ncpus)]
    st = time.time()
    processes = []
    for i in range(ncpus):
        condition1 = np.array([[target_property, 1-target_property]])
        condition2 = np.array([[scaffold_property, 1-scaffold_property]])
        p = mp.Process(target=sample, args=(shared_model, smiles, scaffold,  condition1, condition2, i, retval, args))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        p.join() 
    end = time.time()       
    valid = [j[1] for k in retval for j in k ]
    valid = [v for v in valid if v is not None]
    print ('before remove duplicate : ', len(valid))
    valid = list(set(valid))
    print ('after remove duplicate', len(valid))
    w = open(args.output_filename, 'w')
    w.write(scaffold[0]+'\toriginal\tegfr\n')
    for idx in range(len(valid)):
        w.write(valid[idx] + '\t' + 'gen_' + str(idx) + '\tegfr\n')
    w.close()                




