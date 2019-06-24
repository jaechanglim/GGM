import argparse
import os
from os.path import isfile, join
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from GGM.models.ggm import GGM
from GGM.utils.data import GGMDataset, GGMSampler
import GGM.utils.util as util


def evaluate(shared_model, data, cycle, num_cycles, save_fpath, retval_list, args):
    shared_model = util.initialize_model(shared_model, join(save_fpath, "save_{}_{}.pt".format(*cycle)))
    model = GGM(args)
    model.load_state_dict(shared_model.state_dict())
    model.eval()

    st = time.time()

    with torch.no_grad():
        losses = []
        losses_rec = []
        losses_vae = []
        losses_isomer = []
        losses_predict = []
        for i, batch in enumerate(data):
            id, whole, scaffold, condition, mask = batch
            for idx in range(len(batch)):
                retval = model(whole[idx], scaffold[idx], condition[idx], mask[idx])

                if retval is None:
                    continue

                _, __, loss_rec, loss_vae, loss_isomer, loss_predict, ___ = retval
                loss_rec_beta = args.beta2 * loss_rec
                loss_vae_beta = args.beta1 * loss_vae
                loss = loss_rec_beta + loss_vae_beta + loss_isomer + loss_predict

                losses.append(loss)
                losses_rec.append(loss_rec)
                losses_vae.append(loss_vae)
                losses_isomer.append(loss_isomer)
                losses_predict.append(loss_predict)

        et = time.time()

        loss = sum(losses) / len(losses)
        loss_rec = sum(losses_rec) / len(losses_rec)
        loss_vae = sum(losses_vae) / len(losses_vae)
        loss_isomer = sum(losses_isomer) / len(losses_isomer)
        loss_predict = sum(losses_predict) / len(losses_predict)

        total_cycle = int(cycle[0]) * num_cycles + int(cycle[1])

        print("%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" %(cycle[0], cycle[1], total_cycle, loss, loss_rec,
                                                                 loss_vae, loss_isomer, loss_predict, et-st))

        retval_list.append({"total": loss,
                           "reconstruction": loss_rec,
                           "vae": loss_vae,
                           "isomer": loss_isomer,
                           "predict": loss_predict,
                           "total_cycle": total_cycle})


def cmp_loss_dict(dict1, dict2):
    if dict1["total_cycle"] > dict2["total_cycle"]:
        return 1
    if dict1["total_cycle"] < dict2["total_cycle"]:
        return -1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpus',
                        help='number of cpus',
                        type=int,
                        default=1)
    parser.add_argument("--ncpus_train",
                        help="number of cpus used in training process",
                        type=int,
                        default=1)
    parser.add_argument('--item_per_cycle',
                        help='iteration per cycle set in training process',
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
    parser.add_argument('--beta1',
                        help='beta1: lambda paramter for VAE training',
                        type=float,
                        default=5e-3)
    parser.add_argument('--beta2',
                        help='beta2: lambda paramter for reconstruction '
                             'training',
                        type=float,
                        default=5e-2)
    parser.add_argument('--dropout',
                        help='dropout: dropout rate of property predictor',
                        type=float,
                        default=0.0)
    parser.add_argument('--active_ratio',
                        help='active ratio in sampling (default: no matter)',
                        type=float)
    parser.add_argument('--smiles_path',
                        help='SMILES-data path')
    parser.add_argument("--smiles_path_train",
                        help="SMILES-data path used in training process")
    parser.add_argument('--data_paths',
                        help='property-data paths',
                        nargs='+',
                        default=[],
                        metavar='PATH')
    parser.add_argument('--save_fpath',
                        help='path of a directory of saved models to evaluate')
    args = parser.parse_args()

    depath = lambda path: os.path.realpath(os.path.expanduser(path))
    smiles_path = depath(args.smiles_path)
    smiles_path_train = depath(args.smiles_path_train)
    data_paths = [depath(path) for path in args.data_paths]
    save_fpath = depath(args.save_fpath) if args.save_fpath else None

    save_files = [f for f in os.listdir(save_fpath) if isfile(join(save_fpath, f))]
    cycles = [(file.split(".")[0].split("_")[1], file.split(".")[0].split("_")[2]) for file in save_files]
    cycles_iter = iter(cycles)

    dataset = GGMDataset(smiles_path, *data_paths)
    dataset_train = GGMDataset(smiles_path_train)

    args.N_properties = dataset.N_properties
    args.N_conditions = dataset.N_conditions

    mp.set_start_method('spawn')
    torch.manual_seed(1)

    shared_model = GGM(args)
    shared_model.share_memory()

    num_cycles = int(len(dataset_train) / args.ncpus_train / args.item_per_cycle)

    print("Model #Params: %dK" %
          (sum([x.nelement() for x in shared_model.parameters()]) / 1000,))

    print(f"""\
        ncpus             : {args.ncpus}
        OMP_NUM_THREADS   : {os.environ.get('OMP_NUM_THREADS')}
        Number of data    : {len(dataset)}
        Minibatch size    : {args.item_per_cycle} per CPU per cycle
        Number of cycles  : {num_cycles} per epoch
        dim_of_node_vector: {args.dim_of_node_vector}
        dim_of_edge_vector: {args.dim_of_edge_vector}
        dim_of_FC         : {args.dim_of_FC}
        beta1             : {args.beta1}
        dropout           : {args.dropout}
        SMILES data path  : {smiles_path}
        Data directories  : {data_paths}
        Restart from      : {save_fpath}
        """)

    print("epoch  cyc  totcyc  loss  loss1  loss2  loss3  loss4  time")

    if args.active_ratio is None:
        data = DataLoader(dataset,
                          batch_size=args.item_per_cycle,
                          shuffle=True)
    # Sample active and inactive keys by the required ratio.
    else:
        sampler = GGMSampler(dataset,
                             ratios=args.active_ratio)
        data = DataLoader(dataset,
                          batch_size=args.item_per_cycle,
                          sampler=sampler)

    retval_list = mp.Manager().list()
    processes = []
    loss_list = []
    done = False
    with SummaryWriter() as writer:
        while not done:
            for pid in range(args.ncpus):
                try:
                    cycle = next(cycles_iter)
                    proc = mp.Process(target=evaluate,
                                      args=(shared_model, data, cycle, num_cycles, save_fpath, retval_list, args))
                    proc.start()
                    processes.append(proc)
                    time.sleep(0.1)
                except StopIteration:
                    done = True
                    break
        for proc in processes:
            proc.join()
        retval_list = list(retval_list)
        retval_list = sorted(retval_list, key=lambda x: x["total_cycle"])
        print(retval_list)
        for loss in retval_list:
            writer.add_scalars("loss/loss_group",
                               {"total": loss["total"],
                                "reconstruction": loss["reconstruction"],
                                "vae": loss["vae"],
                                "isomer": loss["isomer"],
                                "predict": loss["predict"]},
                               loss["total_cycle"])



