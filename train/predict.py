import argparse
import os
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from GGM.models.ggm import GGM
from GGM.utils.data import GGMDataset
import GGM.utils.util as util
from tqdm import tqdm


def predict(model, data):
    model.eval()
    losses = []
    whole_pred = []
    whole_true = []
    diffs = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(data)):
            whole = batch[1][0]
            scaffold = batch[2][0]
            conditions = batch[3][0]
            masks = batch[4][0]
            if masks[0].item() != 0:
                whole_condition = model.predict_properties(whole)
                if whole_condition is not None:
                    if (i < 10):
                        print(whole, whole_condition[0][0], conditions[0])
                    criteria = nn.MSELoss()
                    whole_pred.append(whole_condition[0][0])
                    whole_true.append(conditions[0])
                    loss = criteria(whole_condition[0][0], conditions[0])
                    losses.append(loss.item())
                
        np.asarray(whole_pred).astype(np.float64)
        np.asarray(whole_true).astype(np.float64)
    print(sum(losses)/len(losses))
    print(r2_score(whole_pred, whole_true))



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

    predict(model, data)
