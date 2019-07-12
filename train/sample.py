import argparse
import os
import time
import random

import torch
import torch.multiprocessing as mp

import GGM.utils.util as util
from GGM.models.ggm import GGM


def predict(shared_model, id, smiles, retval_dict, args):
    model = GGM(args)
    model.load_state_dict(shared_model.state_dict())
    model.eval()

    with torch.no_grad():
        condition = model.predict_properties(smiles)
        if condition is not None:
            retval_dict[id] = condition.item()


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
    model.eval()
    st1 = time.time()

    with torch.no_grad():
        for idx, (s1, s2) in enumerate(zip(wholes, scaffolds)):
            model.load_state_dict(shared_model.state_dict())
            model.eval()
            retval = model.sample(s1, s2, latent_vector=None, condition=condition,
                                  stochastic=args.stochastic)
            # retval = shared_model(s)
            if retval is None: continue
            predicted_condition = model.predict_properties(retval)
            # Save the given whole SMILES and the new SMILES.
            retval_list[pid].append((s1, retval, predicted_condition.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpus',
                        help='number of cpus',
                        type=int,
                        default=1)
    parser.add_argument("--item_per_cycle",
                        help="number of samples per CPU and molecule",
                        type=int,
                        default=1)
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
    parser.add_argument("--num_scaffolds",
                        help="number of scaffolds to be used",
                        type=int,
                        default=1)
    parser.add_argument("--smiles_path",
                        help="path of file with molecules' id, whole smiles, and scaffold smiles",
                        type=str)
    parser.add_argument("--scaffold_smiles",
                        help="scaffold smiles",
                        type=str)
    parser.add_argument("--scaffold_property",
                        help="scaffold property",
                        type=float)
    parser.add_argument("--data_path",
                        help="path of file with molecules' id, whole property, and scaffold property",
                        type=str)
    parser.add_argument("--min_scaffold_value",
                        help="minimum property value of scaffold to be used",
                        type=float)
    parser.add_argument("--max_scaffold_value",
                        help="maximum property value of scaffold to be used",
                        type=float)
    parser.add_argument("--target_property",
                        help="value of target property",
                        type=float)
    parser.add_argument('--save_fpath',
                        help='file path of saved model',
                        type=str)
    parser.add_argument('--output_filename',
                        help='output file name',
                        type=str)
    parser.add_argument('--stochastic',
                        help='stocahstically add node and edge',
                        action='store_true')
    args = parser.parse_args()

    depath = lambda path: os.path.realpath(os.path.expanduser(path))
    save_fpath = depath(args.save_fpath)
    output_filename = depath(args.output_filename)

    mp.set_start_method('spawn')
    torch.manual_seed(1)

    args.N_conditions = 2
    args.N_properties = 1
    shared_model = GGM(args)
    shared_model.share_memory()
    shared_model = util.initialize_model(shared_model, save_fpath)

    print("Model #Params: %dK" % (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))
    print(f"""\
    ncpus                : {args.ncpus}
    OMP_NUM_THREADS      : {os.environ.get('OMP_NUM_THREADS')}
    Sample per CPU       : {args.item_per_cycle} per CPU
    Num of generations   : {args.ncpus * args.item_per_cycle} per scaffold
                            (Total {args.ncpus * args.item_per_cycle * args.num_scaffolds})
    Model path           : {save_fpath}
    Output path          : {output_filename}
    Smiles path          : {args.smiles_path}
    Data path            : {args.data_path}
    Scaffold value range : {args.min_scaffold_value} ~ {args.max_scaffold_value}
    Target property      : {args.target_property}
    dim_of_node_vector   : {args.dim_of_node_vector}
    dim_of_edge_vector   : {args.dim_of_edge_vector}
    dim_of_FC            : {args.dim_of_FC}
    stochastic           : {args.stochastic}
    """)

    scaffold_property = [args.scaffold_property]
    target_property = [args.target_property]
    scaffolds = [args.scaffold_smiles for i in range(args.item_per_cycle)]
    wholes = [None for i in range(args.item_per_cycle)]
    target_condition = target_property.copy()
    scaffold_condition = scaffold_property.copy()
    target_scaffold_condition = target_condition + scaffold_condition

    retval_list = [mp.Manager().list() for i in range(args.ncpus)]
    st = time.time()
    processes = []

    for pid in range(args.ncpus):
        proc = mp.Process(target=sample,
                          args=(shared_model, wholes, scaffolds, target_scaffold_condition, pid, retval_list, args))
        proc.start()
        processes.append(proc)
        time.sleep(0.1)
    for proc in processes:
        proc.join()

    et = time.time()

    generations = [(j[1], j[2]) for k in retval_list for j in k]
    with open(args.output_filename, 'w') as output:
        output.write("ID SCAFFOLD PROPERTY DIFF\n")
        output.write("{} {:.3f}\n".format(args.scaffold_smiles, scaffold_property[0]))
        for idx, data in enumerate(generations):
            generated_smiles, property = data
            output.write("gen_{} {} {:.3f} {:.3f}\n".format(idx, generated_smiles, property,
                                                   abs(property-scaffold_property[0])))
        output.write("\n")
