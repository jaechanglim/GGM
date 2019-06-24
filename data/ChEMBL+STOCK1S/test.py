import argparse
import os
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GGM.models.ggm import GGM
from GGM.utils.data import GGMDataset
import GGM.utils.util as util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path",
                        help="SMILES-data path")
    parser.add_argument('--data_paths',
                        help='property-data paths',
                        nargs='+',
                        default=[],
                        metavar='PATH')
    parser.add_argument("--save_fpath",
                        help="path of a saved model to load")
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
                        help='dropout: dropout rate of property predictor',
                        type=float,
                        default=0.0)
    args = parser.parse_args()

    depath = lambda path: os.path.realpath(os.path.expanduser(path))
    smiles_path = depath(args.smiles_path)
    data_paths = [depath(path) for path in args.data_paths]
    save_fpath = depath(args.save_fpath)

    # Load data.
    dataset = GGMDataset(smiles_path, *data_paths)

    # Get the number of conditions as a hyperparameter.
    # `args` delivers the hyperparameters to `ggm.ggm`.
    args.N_properties = dataset.N_properties
    args.N_conditions = dataset.N_conditions

    # model
    model = GGM(args)

    # initialize parameters of the model
    initial_epoch, initial_cycle = \
        [int(value)
         for value in re.findall('\d+', os.path.basename(save_fpath))]
    model = util.initialize_model(model, save_fpath)

    data = DataLoader(dataset)

    with open("../../sample_100_5_1.txt") as resultFile:
        with open("../../sample_100_5_result.txt", "w") as newFile:
            for line in resultFile:
                data = line.split("\t")
                if not data[0] == "\n":
                    condition = model.predict_properties(data[0])
                    newFile.write(line[:-1] + "\t" + str(condition.item()) +
                                  "\n")
                else:
                    newFile.write("\n")




