from collections import OrderedDict
from shared_optim import SharedRMSprop, SharedAdam
import torch.optim as optim
from utils import *
from torch.autograd import Variable
from ggm import ggm
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Pool


#c1ccc(Nc2ncnc3c2oc2ccccc23)cc1 CHEMBL149512 Brc4cccc(Nc2ncnc3c1ccccc1oc23)c4 6.130768280269024

def sample(shared_model, smiles, scaffold, condition1, condition2, pid, retval_list):
    #optimizer = optim.Adam(shared_model.parameters(), lr=1e-4)
    model=ggm()
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
            new_s = graph_to_smiles(g_gen, h_gen)
        except:
            new_s = None
        retval_list[pid].append((s1, new_s))
    end1 = time.time()

    #print ('accumulate time', pid, end1-st1)



if __name__ == '__main__':
    mp.set_start_method('spawn')
    torch.manual_seed(1)
    shared_model = ggm()
    shared_model.load_state_dict(torch.load('../GGM3/vaetrain/save_check_24_0.pt'))
    shared_model.share_memory()
    print ("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
    num_epoch = 1
    ncpus = 8
    item_per_cycle = 10
    models = []
    for i in range(ncpus):
        models.append(ggm())
    #    time.sleep(10.0)
    scaffold = ['c1ccc(Nc2ncnc3c2oc2ccccc23)cc1' for i in range(item_per_cycle)]
    smiles = [None for i in range(item_per_cycle)]
    for epoch in range(num_epoch):
        for c in range(1):
        #for c in range(num_cycles):
            retval = mp.Manager().list()
            retval = [mp.Manager().list() for i in range(ncpus)]
            st = time.time()
            processes = []
            for i in range(ncpus):
                p = mp.Process(target=sample, args=(shared_model, smiles, scaffold,  np.array([[1,0]]), np.array([[0.85,0.15]]), i, retval))
                p.start()
                processes.append(p)
                time.sleep(0.1)
            for p in processes:
                p.join() 
            end = time.time()       
            valid = []
            for k in retval:
                #print (len(k))
                for j in k:
                    #print (epoch, c, j[0], j[1])
                    if j[1] is not None:
                        valid.append(j[1])
            print ('before valid', len(valid))
            valid = list(set(valid))
            print ('after valid', len(valid))
            w = open('generated_molecule.txt', 'w')
            w.write('Brc4cccc(Nc2ncnc3c1ccccc1oc23)c4\toriginal\tegfr\n')
            for idx in range(len(valid)):
                w.write(valid[idx] + '\t' + 'gen_' + str(idx) + '\tegfr\n')
            w.close()                




