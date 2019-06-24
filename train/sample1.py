import subprocess
import os


if __name__ == "__main__":
    scaffolds = []
    datas = []
    with open("../data/ChEMBL+STOCK1S/id_smiles_scaffold_100.txt") as \
            scaffoldFile:
        for line in scaffoldFile:
            scaffolds.append(line.split(" ")[2])
    with open("../data/ChEMBL+STOCK1S/data_ChEMBL_scaffold_100.txt") as \
            dataFile:
        for line in dataFile:
            datas.append(line.split(" ")[1])

    for i in range(len(scaffolds)):
        os.environ["OMP_NUM_THREADS"] = "1"
        subprocess.run(["python", "../utils/sample.py",
                        "--ncpus", "10", "--item_per_cycle", "1",
                        "--save_fpath",
                        "../results/20190605T1501/save_8_0.pt",
                        "--scaffold", scaffolds[i], "--target_properties",
                        "8.0", "--scaffold_properties", datas[i],
                        "--output_filename", "../sample_100_5.txt",
                        "--stochastic", "--dropout", "0.5"])
