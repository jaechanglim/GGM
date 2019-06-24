import argparse
import os
import time
from rdkit import Chem
import torch
import torch.multiprocessing as mp
import GGM.utils.util as util
from GGM.models.ggm import GGM


def normalize(v, max_v=None, min_v=None):
    """v -> v' in [0, 1]"""
    if max_v is None and min_v is None:
        return v
    else:
        v = min(max_v, v)
        v = max(min_v, v)
        return (v - min_v) / (max_v - min_v)


def sample(shared_model, wholes, scaffolds, condition, pid, retval_list, args):
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
    # optimizer = optim.Adam(shared_model.parameters(), lr=1e-4)
    model = GGM(args)
    st1 = time.time()

    for idx, (s1, s2) in enumerate(zip(wholes, scaffolds)):
        model.load_state_dict(shared_model.state_dict())
        retval = model.sample(s1, s2, latent_vector=None, condition=condition,
                              stochastic=args.stochastic)
        # retval = shared_model(s)
        if retval is None: continue
        # Save the given whole SMILES and the new SMILES.
        retval_list[pid].append((s1, retval))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpus',
                        help='number of cpus',
                        type=int, default=1)
    parser.add_argument('--item_per_cycle',
                        help='number of generations per CPU',
                        type=int,
                        default=128)
    parser.add_argument('--dim_of_node_vector',
                        help='dimension of node_vector',
                        type=int,
                        default=128)
    parser.add_argument('--dim_of_edge_vector',
                        help='dimension of edge vector',
                        type=int,
                        default=128)
    parser.add_argument('--dim_of_FC',
                        help='dimension of FC',
                        type=int,
                        default=128)
    parser.add_argument('--dropout',
                        help='dropout rate of property predictor',
                        type=float,
                        default=0.0)
    parser.add_argument('--save_fpath',
                        help='file path of saved model',
                        type=str)
    parser.add_argument('--scaffold',
                        help='smiles of scaffold',
                        type=str)
    """
    parser.add_argument('--target_scaffold_properties',
                        help='values of target properties and scaffold properties',
                        nargs='+',
                        default=[],
                        type=float)
    """
    parser.add_argument('--target_properties', help='values of target properties', nargs='+', default=[], type=float)
    parser.add_argument('--scaffold_properties', help='values of scaffold properties', nargs='+', default=[], type=float)
    parser.add_argument('--output_filename',
                        help='output file name',
                        type=str)
    parser.add_argument('--minimum_values',
                        help='minimum values of properties. It will be used for normalization',
                        nargs='+',
                        default=[],
                        type=float)
    parser.add_argument('--maximum_values',
                        help='maximum values of properties. It will be used for normalization',
                        nargs='+',
                        default=[],
                        type=float)
    parser.add_argument('--stochastic',
                        help='stocahstically add node and edge',
                        action='store_true')
    args = parser.parse_args()

    # hyperparameters
    save_fpath = os.path.realpath(os.path.expanduser(args.save_fpath))
    output_filename = os.path.realpath(os.path.expanduser(args.output_filename))

    #target_properties = [normalize(*values) for values in zip(
    #    args.target_properties, args.maximum_values, args.minimum_values)]
    #scaffold_properties = [normalize(*values) for values in zip(
    #    args.scaffold_properties, args.maximum_values, args.minimum_values)]
    target_properties = args.target_properties
    scaffold_properties = args.scaffold_properties
    target_scaffold_properties = target_properties + scaffold_properties

    # lines for multiprocessing
    mp.set_start_method('spawn')
    torch.manual_seed(1)

    # model
    args.N_conditions = len(target_scaffold_properties)
    args.N_properties = len(target_properties)
    shared_model = GGM(args)
    shared_model.share_memory()

    print("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
    print(f"""\
ncpus             : {args.ncpus}
OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS')}
Num of generations: {args.item_per_cycle} per CPU (Total {args.ncpus * args.item_per_cycle})
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

    # initialize parameters of the model
    shared_model = util.initialize_model(shared_model, save_fpath)

    # Copy the same scaffold SMILES for multiple generations.
    scaffolds = [args.scaffold for i in range(args.item_per_cycle)]
    # A whole SMILES can be given and become a latent vector for decoding,
    # but here it is given as None so that a latent is randomly sampled.
    wholes = [None for i in range(args.item_per_cycle)]
    target_condition = target_properties.copy()
    scaffold_condition = scaffold_properties.copy()
    target_scaffold_condition = target_condition + scaffold_condition

    # A list of multiprocessing.managers.ListProxy to collect SMILES
    retval_list = [mp.Manager().list() for i in range(args.ncpus)]
    st = time.time()
    processes = []

    for pid in range(args.ncpus):
        p = mp.Process(target=sample,
                       args=(shared_model, wholes, scaffolds, target_scaffold_condition, pid, retval_list, args))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        p.join()
    end = time.time()

    # retval_list shape -> (ncpus, item_per_cycle, 2)
    generations = [j[1] for k in retval_list for j in k]  # list of new SMILESs

    # Write the generated SMILES strings.
    with open(args.output_filename, 'a') as output:
        output.write(scaffolds[0] + '\toriginal\tscaffold\n')
        for idx, smiles in enumerate(generations):
            output.write(smiles + '\t' + 'gen_' + str(idx) + '\n')
