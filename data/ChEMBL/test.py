import random

if __name__== "__main__":
    """
    ids = []
    with open("./id_smiles_train.txt") as idFile:
        for line in idFile:
            data = line.split(" ")
            ids.append(data[0])

    with open("./data_normalized_train.txt", "w") as newFile:
        with open("./data_normalized.txt") as oldFile:
            for line in oldFile:
                data= line.split(" ")
                if data[0] in ids:
                    newFile.write(line)
    """
    test_ids = []
    train_ids = []

    with open("../test_id_smiles.txt") as idFile:
        for line in idFile:
            data = line.split("\t")
            test_ids.append(data[0])
    with open("../train_id_smiles.txt") as idFile:
        for line in idFile:
            data = line.split("\t")
            train_ids.append(data[0])

    print(train_ids[:10])
    print(test_ids[:10])

    with open("../ChEMBL+STOCK1S/test_STOCK1S.txt", "w") as newFile:
        with open("../affinity/data_normalized.txt") as oldFile:
            for line in oldFile:
                data = line.split("\t")
                newData = [data[0], "None", "None\n"]
                if newData[0] in test_ids:
                    newLine = (" ").join(newData)
                    newFile.write(newLine)

    """
    with open("../ChEMBL+STOCK1S/train_STOCK1S.txt", "w") as newFile:
        with open("../affinity/data_normalized.txt") as oldFile:
            for line in oldFile:
                data = line.split("\t")
                newData = [data[0], "None", "None\n"]
                if newData[0] in train_ids:
                    newLine = (" ").join(newData)
                    newFile.write(newLine)
    """
