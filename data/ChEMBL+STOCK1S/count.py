import random
from tqdm import tqdm

if __name__ == "__main__":
    test_id = []
    train_id = []
    with open("data_STOCK_test_1.txt", "w") as newFile:
        with open("id_smiles_test.txt") as idFile:
            for line in tqdm(idFile, desc = "test_read"):
                data = line.split("\t")
                test_id.append(data[0])

        with open("../logp/data.txt") as dataFile:
            for line in tqdm(dataFile, desc = "test_write"):
                data = line.split("\t")
                id = data[0]
                scaffold = data[2]
                if id in test_id:
                    rand = random.random()
                    if rand < 0.01:
                        newFile.write(id + "\tNone\t" + scaffold)
                    else:
                        newFile.write(line)

    with open("data_STOCK_train_1.txt", "w") as newFile:
        with open("id_smiles_train.txt") as idFile:
            for line in tqdm(idFile, desc = "train_read"):
                data = line.split("\t")
                train_id.append(data[0])

        with open("../logp/data.txt") as dataFile:
            for line in tqdm(dataFile, desc = "train_write"):
                data = line.split("\t")
                id = data[0]
                scaffold = data[2]
                if id in train_id:
                    rand = random.random()
                    if rand < 0.01:
                        newFile.write(id + "\tNone\t" + scaffold)
                    else:
                        newFile.write(line)
