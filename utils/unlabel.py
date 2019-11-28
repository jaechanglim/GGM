import argparse
import os
import random
from tqdm import tqdm

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_smiles_path",
                        help="test SMILES-data path",
                        type=str)
    parser.add_argument("--train_smiles_path",
                        help="train SMILES-data path",
                        type=str)
    parser.add_argument("--data_path",
                        help="path for default data file",
                        type=str)
    parser.add_argument("--new_test_path",
                        help="path for new test file",
                        type=str)
    parser.add_argument("--new_train_path",
                        help="path for new train file",
                        type=str)
    parser.add_argument("--unlabel_ratio",
                        help="unlabeled ratios of property",
                        type=float)
    args = parser.parse_args()

    depath = lambda path: os.path.realpath(os.path.expanduser(path))
    test_smiles_path = depath(args.test_smiles_path)
    train_smiles_path = depath(args.train_smiles_path)
    data_path = depath(args.data_path)
    new_test_path = depath(args.new_test_path)
    new_train_path = depath(args.new_train_path)

    test_id = []
    train_id = []
    with open(args.new_test_path, "w") as newFile:
        with open(args.test_smiles_path) as idFile:
            for line in tqdm(idFile, desc="test_read"):
                data = line.split("\t")
                test_id.append(data[0])

        with open(args.data_path) as dataFile:
            for line in tqdm(dataFile, desc="test_write"):
                data = line.split("\t")
                id = data[0]
                if id in test_id:
                    rand = random.random()
                    if rand < args.unlabel_ratio:
                        newFile.write(id + "\tNone\tNone\n")
                    else:
                        newFile.write(line)

    with open(args.new_train_path, "w") as newFile:
        with open(args.train_smiles_path) as idFile:
            for line in tqdm(idFile, desc="train_read"):
                data = line.split("\t")
                train_id.append(data[0])

        with open(args.data_path) as dataFile:
            for line in tqdm(dataFile, desc="train_write"):
                data = line.split("\t")
                id = data[0]
                scaffold = data[2]
                if id in train_id:
                    rand = random.random()
                    if rand < args.unlabel_ratio:
                        newFile.write(id + "\tNone\tNone\n")
                    else:
                        newFile.write(line)