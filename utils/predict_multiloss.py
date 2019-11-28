import argparse
import os
import time
import random

import subprocess
from multiprocessing import Pool

import torch
import numpy as np
import math

import GGM.utils.util as util
from GGM.models.ggm import GGM


def run(keys):
    smiles_path, data_path, save_dir, epoch, test_smiles_path, test_data_path, existing_result, dropout, output_filename = keys

    command = "OMP_NUM_THREADS=1 \
               python ~/msh/GGM/utils/predict_loss.py \
               --smiles_path {} \
               --data_path {} \
               --save_fpath {}/save_{}_0.pt \
               --test_smiles_path {} \
               --test_data_path {} \
               --existing_result {}\
               --dropout {}\
               --use_subscaffold".format(smiles_path, data_path, save_dir, epoch, test_smiles_path, test_data_path, existing_result, dropout)

    output = subprocess.check_output(command, shell=True).decode("utf-8")
    old_output = output.split("\n")
    final_output = [item for item in old_output if "it/s]" not in item]

    with open(output_filename, "a") as writeFile:
        for item in final_output:
            if "TRAIN" in item:
                epoch = "000" + str(epoch)
                writeFile.write("EPOCH: {}\n".format(epoch[len(epoch)-3:len(epoch)]))
            writeFile.write(item + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpus',
                        help='number of cpus',
                        type=int,
                        default=1)
    parser.add_argument('--smiles_path',
                        help='SMILES-data path')
    parser.add_argument("--data_path",
                        help="path of file with molecules' id, whole property, and scaffold property",
                        type=str)
    parser.add_argument('--save_dir',
                        help='path of a saved model to load')
    parser.add_argument('--max_epoch',
                        help='maximum epoch number for load',
                        type=int,
                        default=25)
    parser.add_argument("--test_smiles_path",
                        help="SMILES-data path")
    parser.add_argument("--test_data_path",
                        help="path of file with molecules' id, whole property, and scaffold property",
                        type=str)
    parser.add_argument('--epoch_interval',
                        help='interval between epochs',
                        type=int,
                        default=1)
    parser.add_argument('--dropout',
                        help='dropout rate of property predictor',
                        type=float,
                        default=0.0)
    parser.add_argument('--output_filename',
                        help='output file name',
                        type=str)
    parser.add_argument('--existing_result',
                        help='whether using existing result or not',
                        type=bool)
    args = parser.parse_args()

    depath = lambda path: os.path.realpath(os.path.expanduser(path))
    save_dir = depath(args.save_dir)
    output_filename = depath(args.output_filename)
    smiles_path = depath(args.smiles_path)
    data_path = depath(args.data_path)
    test_smiles_path = depath(args.test_smiles_path)
    test_data_path = depath(args.test_data_path)

    epochs = [i for i in range(args.max_epoch) if i % args.epoch_interval == 0]
    key_list = [[
        smiles_path, data_path, args.save_dir, epoch, test_smiles_path, test_data_path, args.existing_result, args.dropout, output_filename
    ] for epoch in epochs]

    st = time.time()
    pool = Pool(args.ncpus)
    r = pool.map(run, key_list[:])
    pool.close()
    pool.join()
    end = time.time()
    print("time: ", end - st)

    data = []
    with open(output_filename) as baseFile:
        for line in baseFile:
            if "\n" != line:
                data.append(line)

    whole_data = []
    tmp_data = []
    for idx, item in enumerate(data):
        if "EPOCH" in item and tmp_data != []:
            whole_data.append("".join(tmp_data))
            tmp_data = []
        tmp_data.append(item)
    whole_data.append("".join(tmp_data))
    whole_data.sort()
    with open(output_filename, "w") as newFile:
        for idx, item in enumerate(whole_data):
            if idx != 0:
                newFile.write("\n")
            newFile.write(item)

