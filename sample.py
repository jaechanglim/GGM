import argparse
from collections import OrderedDict
import os
import time

import numpy as np
from rdkit import Chem
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
    """v -> v' in [0, 1]"""
    v = min(max_v, v)
    v = max(min_v, v)
    return (v-min_v)/(max_v-min_v)

def sample(shared_model, wholes, scaffolds, condition1, condition2, pid, retval_list, args):
    """\
    Target function for the multiprocessed sampling.

    Sampled SMILESs are collected by `retval_list`.

    Parameters
    ----------
    shared_model: torch.nn.Module
        A shared trained model to be used in the sampling.
    wholes: list[str] | list[None]
        A list of whole SMILESs to be used as latent vectors.
    scaffolds: list[str]
        A list of scaffold SMILESs from which new SMILESs will be sampled.
    condition1: list[float]
        A list of target property values.
    condition2: list[float]
        A list of scaffold property values.
    pid: int
        CPU index.
    retval_list: list[multiprocessing.managers.ListProxy]
        A list of lists to collect sampled SMILESs.
        In each sampling a given whole SMILES and a sampled SMILES are saved,
        so the final shape will be:
            (ncpus, num_of_generations_per_cpu, 2)
    args: argparse.Namespace
        Delivers hyperparameters from command arguments to the model.
    """
    #optimizer = optim.Adam(shared_model.parameters(), lr=1e-4)
    model=ggm(args)
    st1 = time.time()

    for idx, (s1,s2) in enumerate(zip(wholes, scaffolds)):
        model.load_state_dict(shared_model.state_dict())
        retval = model.sample(s1, s2, latent_vector=None, condition1=condition1, condition2=condition2, stochastic=args.stochastic)
        #retval = shared_model(s)
        if retval is None: continue
        # Save the given whole SMILES and the new SMILES.
        retval_list[pid].append((s1, retval))
    end1 = time.time()
    #print ('accumulate time', pid, end1-st1)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--ncpus', help = 'number of cpus', type = int, default = 1) 
    parser.add_argument('--item_per_cycle', help = 'number of generations per CPU', type = int, default = 128) 
    parser.add_argument('--dim_of_node_vector', help = 'dimension of node_vector', type = int, default = 128) 
    parser.add_argument('--dim_of_edge_vector', help = 'dimension of edge vector', type = int, default = 128) 
    parser.add_argument('--dim_of_FC', help = 'dimension of FC', type = int, default = 128) 
    parser.add_argument('--save_fpath', help = 'file path of saved model', type = str) 
    parser.add_argument('--scaffold', help = 'smiles of scaffold', type = str) 
    parser.add_argument('--target_properties', help='values of target properties', nargs='+', default=[], type=float) 
    parser.add_argument('--scaffold_properties', help='values of scaffold properties', nargs='+', default=[], type=float) 
    parser.add_argument('--output_filename', help = 'output file name', type = str) 
    parser.add_argument('--minimum_values', help='minimum values of properties. It will be used for normalization', nargs='+', default=[], type=float) 
    parser.add_argument('--maximum_values', help = 'maximum values of properties. It will be used for normalization', nargs='+', default=[], type=float) 
    parser.add_argument('--stochastic', help = 'stocahstically add node and edge', action='store_true') 
    args = parser.parse_args()
    
    #hyperparameters
    save_fpath = os.path.realpath(os.path.expanduser(args.save_fpath))
    output_filename = os.path.realpath(os.path.expanduser(args.output_filename))

    # Normalize the property values to be in [0, 1].
    target_properties = [normalize(*values) for values in zip(args.target_properties, args.maximum_values, args.minimum_values)]
    scaffold_properties = [normalize(*values) for values in zip(args.scaffold_properties, args.maximum_values, args.minimum_values)]

    #lines for multiprocessing
    mp.set_start_method('spawn')
    torch.manual_seed(1)

    #model 
    args.N_conditions = len(target_properties) + len(scaffold_properties)
    shared_model = ggm(args)
    shared_model.share_memory()

    print ("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
    print(f"""\
ncpus             : {args.ncpus}
OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS')}
Num of generations: {args.item_per_cycle} per CPU (Total {args.ncpus*args.item_per_cycle})
Model path        : {save_fpath}
Output path       : {output_filename}
Scaffold          : {args.scaffold}
Scaffold values   : {args.scaffold_properties} -> {scaffold_properties}
Target values     : {args.target_properties} -> {target_properties}
dim_of_node_vector: {args.dim_of_node_vector}
dim_of_edge_vector: {args.dim_of_edge_vector}
dim_of_FC         : {args.dim_of_FC}
stochastic        : {args.stochastic}
""")
    
    #initialize parameters of the model 
    shared_model = utils.initialize_model(shared_model, save_fpath)
    
    # Copy the same scaffold SMILES for multiple generations.
    scaffolds = [args.scaffold for i in range(args.item_per_cycle)]
    # A whole SMILES can be given and become a latent vector for decoding,
    # but here it is given as None so that a latent is randomly sampled.
    wholes = [None for i in range(args.item_per_cycle)]
    condition1 = target_properties.copy()
    condition2 = scaffold_properties.copy()
    
    # A list of multiprocessing.managers.ListProxy to collect SMILES
    retval_list = [mp.Manager().list() for i in range(args.ncpus)]
    st = time.time()
    processes = []
    
    for pid in range(args.ncpus):
        p = mp.Process(target=sample, args=(shared_model, wholes, scaffolds, condition1, condition2, pid, retval_list, args))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        p.join() 
    end = time.time()       

    # retval_list shape -> (ncpus, item_per_cycle, 2)
    generations = [j[1] for k in retval_list for j in k]  # list of new SMILESs

    # Write the generated SMILES strings.
    with open(args.output_filename, 'w') as output:
        output.write(scaffolds[0]+'\toriginal\tscaffold\n')
        for idx, smiles in enumerate(generations):
            output.write(smiles + '\t' + 'gen_' + str(idx) + '\n')
