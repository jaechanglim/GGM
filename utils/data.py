import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


class GGMSampler(Sampler):

    def __init__(self, dataset, ratios=None, replacement=True):
        self.indicies = list(range(len(dataset)))
        self.replacement = replacement
        num_properties = dataset.N_properties

        if ratios is None:
            ratios = [[1, 1] for _ in range(num_properties)]
            labels = list(range(2 ** num_properties))
            label_to_ratio = [1.0] * len(labels)
        else:
            label_to_ratio = [[ratios, 1-ratios] for _ in range(num_properties)]
            labels = list(range(2 ** num_properties))
        label_to_count = [0] * len(labels)
        id_to_label = {}
        for id, _, __, condition, mask in dataset:
            label = 0
            for i in range(num_properties):
                if mask[i] == 0:
                    label += 2 ** i * 0
                else:
                    label += 2 ** i * 1
            label_to_count[label] += 1
            id_to_label[id] = label

        print(label_to_count)

        self.min_count = min(label_to_count)
        self.max_count = max(label_to_count)

        label_to_weight = \
            [1.0 / count if count != 0 else 0 for count in
             label_to_count]
        for label, ratio in enumerate(label_to_ratio):
            label_to_weight[label*2] *= ratio[0]
            label_to_weight[label*2+1] *= ratio[1]
        self.weights = [label_to_weight[id_to_label[id]]
                        for id in dataset.id_list]
        self.weights = torch.tensor(self.weights, dtype=torch.double)

    def __iter__(self):
        """
        return (self.indicies[i] for i in torch.multinomial(self.weights,
                                                            self.num_samples,
                                                            self.replacement))
        """
        return iter(torch.multinomial(self.weights,
                                      2 * self.min_count,
                                      self.replacement).tolist())

    def __len__(self):
        return self.num_samples



class GGMBoundarySampler(Sampler):
    """
    ratios: [prop1:[None, Low, High], prop2: [None, Low, High] ... ]
    """

    def __init__(self, dataset, num_samples, ratios=None, boundaries=.5,
                 replacement=False):
        self.indicies = list(range(len(dataset)))
        self.num_samples = num_samples
        self.replacement = replacement
        num_properties = dataset.N_properties

        if ratios is None:
            ratios = [[1, 1, 1] for _ in range(num_properties)]

        if isinstance(boundaries, float):
            boundaries = [boundaries for _ in range(num_properties)]

        labels = list(range(3 ** num_properties))

        label_to_ratio = [1.0] * len(labels)
        if self._is_affinity(ratios):
            if num_properties == 1:
                label_to_ratio = [1, 0, 0]
            elif num_properties == 2:
                label_to_ratio = [1, 1, 1, 1, 0, 0, 1, 0, 0]
            elif num_properties == 3:
                label_to_ratio = \
                    [1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 0, 0, 1, 0, 0,
                     1, 1, 1, 1, 0, 0, 1, 0, 0]
            else:
                raise NotImplementedError
        else:
            if num_properties == 1:
                label_to_ratio = ratios[0]
            elif num_properties == 2:
                for i in range(3):
                    for j in range(3):
                        label_to_ratio[3*i+j] *= ratios[0][i] * ratios[1][j]
            elif num_properties == 3:
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            label_to_ratio[9*i+3*j+k] *= \
                                ratios[0][i] * ratios[1][j] * ratios[2][k]
            else:
                raise NotImplementedError


        """
        for i in range(len(label_to_ratio)):
            prob = 1
            for j in range(num_properties):
                ratio = self.ratios[j] if i//(2**j)%2==1 else 1-self.ratios[j]
                prob *= ratio
            self.label_to_ratio[i] = prob
        """

        label_to_count = [0] * len(labels)
        id_to_label = {}
        for id, _, __, condition, mask in dataset:
            label = 0
            for i in range(num_properties):
                if mask[i] == 0:
                    label += 3 ** i * 0
                else:
                    if condition[i] > boundaries[i]:
                        label += 3 ** i * 1
                    else:
                        label += 3 ** i * 2
            label_to_count[label] += 1
            id_to_label[id] = label

        print(label_to_count)

        label_to_weight = \
            [1.0/count if count != 0 else 0 for count in
             label_to_count]
        for label, ratio in enumerate(label_to_ratio):
            label_to_weight[label] *= ratio

        self.weights = [label_to_weight[id_to_label[id]]
                        for id in dataset.id_list]
        self.weights = torch.tensor(self.weights, dtype=torch.double)

    def _get_label(self, id, label_to_ids):
        for label, ids in enumerate(label_to_ids):
            if id in ids:
                return label

    def _is_affinity(self, ratios):
        if ratios is None:
            return False
        for ratio in ratios:
            if int(ratio[0]) == 0:
                return False
        return True

    def __iter__(self):
        """
        return (self.indicies[i] for i in torch.multinomial(self.weights,
                                                            self.num_samples,
                                                            self.replacement))
        """
        return iter(torch.multinomial(self.weights,
                                      self.num_samples,
                                      self.replacement).tolist())

    def __len__(self):
        return self.num_samples


class GGMDataset(Dataset):

    def __init__(self, smiles_path, *data_paths):

        super(GGMDataset, self).__init__()

        self.N_properties = 0

        data = self.load_data(smiles_path, data_paths)
        self.id_list = data[0]
        self.id_to_smiles = data[1]
        self.id_to_conditions = data[2]
        self.id_to_mask = data[3]

        self.N_conditions = 2 * self.N_properties

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]
        whole_smiles, scaffold_smiles = self.id_to_smiles[id]
        condition = np.float32(self.id_to_conditions[id])
        mask = np.float32(self.id_to_mask[id])
        return id, \
               whole_smiles, \
               scaffold_smiles, \
               condition, \
               mask

    def load_data(self, smiles_path, *data_paths):
        """\
        Read data text files.

        The structure of `smiles_path` should be

            #    whole      scaffold
            ID1  SMILES1_1  SMILES1_2
            ID2  SMILES2_1  SMILES2_2
            ...

        And the structure of EACH path in `data_paths` should be

            #    whole               scaffold
            ID1  value_of_SMILES1_1  value_of_SMILES1_2
            ID2  value_of_SMILES2_1  value_of_SMILES2_2
            ...

        If some conditions are missing, the value must be None

            #    whole               scaffold
            ID1  value_of_SMILES1_1  None
            ID2  None                value_of_SMILES2_2
            ...

        Parameters
        ----------
        smiles_path: str
            A data text path of IDs and SMILESs.
        data_paths: iterable of str
            Each is a data text path of IDs and property values.

        Returns
        -------
        id_to_smiles: dict[str, list[str]]
        id_to_whole_conditions: dict[str, list[float]]
            { ID1: [whole_value_of_property1, whole_value_of_property2, ...], ... }
        id_to_scaffold_conditions: dict[str, list[float]]
            { ID1: [scaffold_value_of_property1, scaffold_value_of_property2, ...], ... }
        """
        # {id:[whole_smiles, scaffold_smiles], ...}
        id_to_smiles = self.dict_from_txt(smiles_path)
        data_dicts = [self.dict_from_txt(path, float) for path in data_paths[0]]
        self.N_properties = len(data_dicts)

        id_list = list(id_to_smiles.keys())

        # Collect condition values of multiple properties.
        id_to_conditions = {}
        id_to_mask = {}
        for id in id_list:
            whole_conditions = [0.0] * self.N_properties
            scaffold_conditions = [0.0] * self.N_properties
            whole_mask = [0] * self.N_properties
            scaffold_mask = [0] * self.N_properties
            for i, data_dict in enumerate(data_dicts):
                if id in data_dict.keys():
                    if data_dict[id][0] is not None:
                        whole_conditions[i] = data_dict[id][0]
                        whole_mask[i] = 1
                    if data_dict[id][1] is not None:
                        scaffold_conditions[i] = data_dict[id][1]
                        scaffold_mask[i] = 1
            id_to_conditions[id] = whole_conditions + scaffold_conditions
            id_to_mask[id] = whole_mask + scaffold_mask
        return id_list, \
               id_to_smiles, \
               id_to_conditions, \
               id_to_mask

    def dict_from_txt(self, path, dtype=None):
        """
        Generate a dict from a text file.

        The structure of `path` should be

            key1  value1_1  value1_2  ...
            key2  value2_1  value2_2  ...
            ...

        and then the returned dict will be like

            { key1:[value1_1, value1_2, ...], ... }

        NOTE that the dict value will always be a list,
        even if the number of elements is less than 2.

        Parameters
        ----------
        path: str
            A data text path.
        dtype: type | None
            The type of values (None means str).
            Keys will always be str type.

        Returns
        -------
        out_dict: dict[str, list[dtype]]
        """
        out_dict = {}
        # np.genfromtxt or np.loadtxt are slower than pure Python!
        with open(path) as f:
            for line in f:
                row = line.split()
                values = []
                for value in row[1:]:
                    if value == "None":
                        values.append(None)
                    else:
                        if dtype is None:
                            values.append(value)
                        else:
                            values.append(dtype(value))
                out_dict[row[0]] = values
        return out_dict


if __name__ == "__main__":
    ts = time.time()
    # dataset = GGMDataset("../data/ChEMBL+STOCK1S/id_smiles_train.txt",
    #                      "../data/ChEMBL+STOCK1S/data_train.txt")
    dataset = GGMDataset("../data/ChEMBL+STOCK1S/id_smiles_STOCK_train.txt",
                         "../data/ChEMBL+STOCK1S/data_STOCK_train_1.txt")
    sampler = GGMSampler(dataset, 0.91)
    data = DataLoader(dataset,
                      batch_size=100000,
                      sampler=sampler)
    te = time.time()

    sampled_batch = None
    for batch in data:
        sampled_batch = batch
        break

    label_to_count = [0] * (2 ** 1)

    for i in range(len(sampled_batch[0])):
        label = 0
        for prop in range(1):
            value = sampled_batch[3][i][prop].item()
            if not value < 0.00000001:
                label += 2 ** prop * 1
        label_to_count[label] += 1

    print(label_to_count)

