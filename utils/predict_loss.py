import argparse
import os
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from GGM.models.ggm import GGM
from GGM.utils.data import GGMDataset, GGMSampler
import GGM.utils.util as util
from tqdm import tqdm


def predict(model, data, existing_result, is_train=None):
    model.eval()
    whole_pred = []
    whole_true = []
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    tb_losses = []

    # _, wholes, scaffolds, conditions, masks = data
    # zipped_data = list(zip(wholes, scaffolds, conditions, masks))

    with torch.no_grad():
        for i, single_data in tqdm(enumerate(data)):
            whole = single_data[1][0]
            conditions = single_data[3][0]
            masks = single_data[4][0]
            if masks[0].item() != 0:
                whole_condition = model.predict_properties(whole)
                if whole_condition is not None:
                    whole_pred.append(whole_condition[0][0])
                    whole_true.append(conditions[0])
                    tb_loss = torch.sqrt(mse(whole_condition[0][0], conditions[0]))
                    tb_losses.append(tb_loss)

        np.asarray(whole_pred).astype(np.float64)
        np.asarray(whole_true).astype(np.float64)
        np.asarray(tb_losses).astype(np.float64)

    total_tb_loss = np.mean(tb_losses) / torch.sqrt(torch.tensor(2.0))

    maeloss = mae(torch.FloatTensor(whole_pred), torch.FloatTensor(whole_true))
    mseloss = mse(torch.FloatTensor(whole_pred), torch.FloatTensor(whole_true))
    r2score = r2_score(whole_true, whole_pred)

    if is_train is None:
        print("MAE loss: {:.4f}, MSE loss: {:.4f}, r2 score: {:.4f}".format(maeloss, mseloss, r2score))
        if existing_result:
            print("tensorbaord MAE loss: {:.4f}".format(total_tb_loss))
        return

    if is_train:
        print("TRAIN SET\nMAE loss: {:.4f}, MSE loss: {:.4f}, r2 score: {:.4f}".format(maeloss, mseloss, r2score))
    else:
        print("TEST SET\nMAE loss: {:.4f}, MSE loss: {:.4f}, r2 score: {:.4f}".format(maeloss, mseloss, r2score))
        if existing_result:
            print("tensorbaord MAE loss: {:.4f}".format(total_tb_loss))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path",
                        help="SMILES-data path")
    parser.add_argument("--data_path",
                        help="path of file with molecules' id, whole property, and scaffold property",
                        type=str)
    parser.add_argument("--test_smiles_path",
                        help="SMILES-data path")
    parser.add_argument("--test_data_path",
                        help="path of file with molecules' id, whole property, and scaffold property",
                        type=str)
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
    parser.add_argument('--existing_result',
                        help='whether using existing result or not',
                        type=bool)
    parser.add_argument('--use_subscaffold',
                        help='whether use subscaffold or not for training model',
                        action='store_true' )
    args = parser.parse_args()

    depath = lambda path: os.path.realpath(os.path.expanduser(path))
    smiles_path = depath(args.smiles_path)
    data_path = depath(args.data_path)
    save_fpath = depath(args.save_fpath)
    dataset = GGMDataset(smiles_path, data_path)
    # Get the number of conditions as a hyperparameter.
    # `args` delivers the hyperparameters to `ggm.ggm`.
    args.N_properties = dataset.N_properties
    args.N_conditions = dataset.N_conditions

    data = DataLoader(dataset)

    # model
    model = GGM(args)
    model = util.initialize_model(model, save_fpath)

    if args.test_smiles_path:
        test_smiles_path = depath(args.test_smiles_path)
        test_data_path = depath(args.test_data_path)
        test_dataset = GGMDataset(test_smiles_path, test_data_path)
        test_data = DataLoader(test_dataset)
        predict(model, data, args.existing_result, True)
        predict(model, test_data, args.existing_result, False)
    else:
        predict(model, data, args.existing_result)
