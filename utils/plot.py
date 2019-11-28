import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re


def select(datalist):
    rt = []
    for data in datalist:
        _, value = data.split(": ")
        rt.append(value)
    return rt


def plot(epoch_list, train_data, test_data, type, save_dir):
    df = pd.DataFrame({"epoch": epoch_list * 2,
                       type: train_data + test_data,
                       "type": ["train" for i in range(len(train_data))] + ["test" for i in range(len(test_data))]},
                      dtype=float)

    plot = sns.lineplot(x="epoch", y=type, hue="type", data=df)
    if type == "r2":
        plot.set(ylim=(None, 1))
    else:
        plot.set(ylim=(0, None))
    fig = plot.get_figure()
    fig.savefig(save_dir[:-4] + "_" + type + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path",
                        help="file path",
                        type=str)
    parser.add_argument("--value_type",
                        help="type of values want to check: mae, mse, r2",
                        type=str)
    args = parser.parse_args()

    depath = lambda path: os.path.realpath(os.path.expanduser(path))
    file_path = depath(args.file_path)

    data = []
    with open(file_path) as baseFile:
        for line in baseFile:
            if "\n" != line:
                data.append(line)

    whole_data = []
    tmp_data = []
    epoch_list = []
    for idx, item in enumerate(data):
        if "EPOCH" in item:
            epoch_list.append(item.split(": ")[1])
            if tmp_data:
                whole_data.append("".join(tmp_data))
                tmp_data = []
        tmp_data.append(item)
    whole_data.append("".join(tmp_data))

    train_data = []
    test_data = []
    for item in whole_data:
        lines = item.split("\n")
        train_data.append(select(lines[2].split(", ")))
        test_data.append(select(lines[4].split(", ")))

    train_mae, train_mse, train_r2 = zip(*train_data)
    test_mae, test_mse, test_r2 = zip(*test_data)

    if args.value_type == "mae":
        train_data, test_data = train_mae, test_mae
    elif args.value_type == "mse":
        train_data, test_data = train_mse, test_mse
    elif args.value_type == "r2":
        train_data, test_data = train_r2, test_r2
    plot(epoch_list, train_data, test_data, args.value_type, file_path)
