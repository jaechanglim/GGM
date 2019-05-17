import random

if __name__== "__main__":
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