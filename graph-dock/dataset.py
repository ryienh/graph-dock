"""
Implements data loading and pre-processing for chemical docking pre-training
    Currently, this only supports SMILES formatted data (such as the D4 dataset used in the experiment).
    The only front facing function is get_train_val_test_loaders(...), which should be called from the
    training script.

For questions or comments, contact rhosseini@anl.gov
"""

import numpy as np
import pandas as pd
import tqdm
import os
from pysmiles import read_smiles
from mendeleev import element

import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from utils import get_config, suppress_stdout_stderr


def get_train_val_test_loaders(batch_size, transform=None):

    tr, va, te = get_train_val_test_dataset(transform)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, pin_memory=False)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, pin_memory=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, pin_memory=False)


    return tr_loader, va_loader, te_loader


def get_train_val_test_dataset(transform=None):
    root = os.path.join(get_config("data_dir"), get_config("dataset_id"))
    tr = ChemDataset(root, "train", transform=transform)
    va = ChemDataset(root, "val", transform=transform)
    te = ChemDataset(root, "test", transform=transform)

    return tr, va, te


class ChemDataset(InMemoryDataset):
    def __init__(self, root, partition, transform=None, pre_transform=None):
        """
        Reads in necessary data from disk.
        """

        if partition not in ["train", "val", "test"]:
            raise ValueError("Partition {} does not exist".format(partition))

        self.partition = partition

        super().__init__(root, transform, pre_transform)

        if partition == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])

        elif partition == "val":
            self.data, self.slices = torch.load(self.processed_paths[1])

        elif partition == "test":
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return ["d4_dock_data_50k.csv"]

    @property
    def processed_file_names(self):
        # Note: currently processess all partitions if one is missing
        return ["train.pt", "val.pt", "test.pt"]

    # TODO: implement auto download of dataset
    # def download(self):

    def process(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(0)
        self.data = pd.read_csv(get_config("clean_data_path"))

        # check for dataset version
        ds_version = get_config("dataset_id")
        if ds_version.endswith("v0.2"):
            data_list = self._load_data_v_0_2()
        elif ds_version.endswith("v0.3"):
            data_list = self._load_data_v_0_3()
        elif ds_version.endswith("v0.4"):
            data_list = self._load_data_v_0_4()
        else:
            raise ValueError(
                f"Dataset version {ds_version} is not supported. Please update config file and try again."
            )

        data, slices = self.collate(data_list)

        if self.partition == "train":
            torch.save((data, slices), self.processed_paths[0])
        elif self.partition == "val":
            torch.save((data, slices), self.processed_paths[1])
        elif self.partition == "test":
            torch.save((data, slices), self.processed_paths[2])

    def _load_data_v_0_2(self):
        """
        Loads a single data partition from file to memory.
        Scales labels to zero mean and unit s.d.
        Stores element type in one-hot encoding with length 10.
        Datapoints with NaN labels are dropped.
        """

        # TODO: port to pygeo system of transform, pre_transform, pre_filter

        # select appropriate partition
        df = self.data[self.data.partition == self.partition]

        # store dictionary of chem names -> graphs #TODO: implement this for inference
        X = df["smiles"]
        y = df["dockscore"]
        X = X.to_numpy()
        y = y.to_numpy()

        # drop NaN (prev fill NaN values with minimum)
        X = X[~np.isnan(y)]
        y = y[~np.isnan(y)]

        # scale labels
        y = (y - np.mean(y)) / np.sqrt(np.var(y))

        # convert data to graphs
        X = self._smiles_2_graph(X)

        assert len(X) == len(y)

        for graph, label in zip(X, y):
            graph.y = torch.from_numpy(np.asarray(label))

        return X

    def _load_data_v_0_3(self):
        """
        Loads a single data partition from file to memory.
        Scales labels to zero mean and unit s.d., then clips positive results to 0, adds NaNs as 0.
        Stores element type in one-hot encoding with length 10.
        """

        # select appropriate partition
        df = self.data[self.data.partition == self.partition]

        # store dictionary of chem names -> graphs #TODO: implement this for inference
        X = df["smiles"]
        y = df["dockscore"]
        X = X.to_numpy()
        y = y.to_numpy()
        y_raw = y.copy()

        # scale labels
        y[~np.isnan(y)] = (y[~np.isnan(y)] - np.mean(y[~np.isnan(y)])) / np.sqrt(
            np.var(y[~np.isnan(y)])
        )

        # clip labels to (max) TODO: check this with Austin
        max_scaled_label = np.max(y[y_raw <= 0])
        y[y_raw > 0] = max_scaled_label

        # add NaNs as 0
        y[np.isnan(y)] = max_scaled_label

        # convert data to graphs
        X = self._smiles_2_graph(X)

        assert len(X) == len(y)

        for graph, label in zip(X, y):
            graph.y = torch.from_numpy(np.asarray(label))

        return X

    def _load_data_v_0_4(self):
        """
        Loads a single data partition from file to memory.
        Scales labels to zero mean and unit s.d., then clips positive results to 0, adds NaNs as 0. Flips labels.
        Stores element type in one-hot encoding with length 10.
        """

        # select appropriate partition
        df = self.data[self.data.partition == self.partition]

        # store dictionary of chem names -> graphs #TODO: implement this for inference
        X = df["smiles"]
        y = df["dockscore"]
        X = X.to_numpy()
        y = y.to_numpy()
        y_raw = y.copy()

        # scale labels
        y[~np.isnan(y)] = (y[~np.isnan(y)] - np.mean(y[~np.isnan(y)])) / np.sqrt(
            np.var(y[~np.isnan(y)])
        )

        # flip labels
        y = -1 * y

        # clip labels to (max) TODO: check this with Austin
        min_scaled_label = np.min(y[y_raw <= 0])
        y[y_raw > 0] = min_scaled_label

        # add NaNs as 0
        y[np.isnan(y)] = min_scaled_label

        # convert data to graphs
        X = self._smiles_2_graph(X)

        assert len(X) == len(y)

        for graph, label in zip(X, y):
            graph.y = torch.from_numpy(np.asarray(label))

        return X

    def _smiles_2_graph(self, smiles_list):
        """
        Converts list of smiles strings to Pytorch geometric graphs
        """
        X = []
        for molecule in tqdm.tqdm(smiles_list):

            # suppresses warnings regarding stereochemistry info
            with suppress_stdout_stderr():
                x = read_smiles(
                    molecule,
                    explicit_hydrogen=False,
                    zero_order_bonds=True,
                    reinterpret_aromatic=True,
                )

            # cycle through network and preprocess TODO: optimize
            for idx in range(len(list(x.nodes))):

                # represent element by one hot encoded vector (previously by atomic number)
                elt = np.zeros(10)  # 9 elts found in d4_50k_v0.2 train set, 1 for other
                atomic_num = element(x.nodes[idx]["element"]).atomic_number
                elt[self._get_element_one_hot_index(atomic_num)] = 1
                x.nodes[idx]["element"] = elt

                # cast aromatic bool to int
                x.nodes[idx]["aromatic"] = float(int(x.nodes[idx]["aromatic"]))

                # remove stereochemical information - FIXME: discuss sols
                try:
                    del x.nodes[idx]["stereo"]
                except Exception:
                    pass

            # convert to torch and append to list
            x = from_networkx(
                x, group_node_attrs=["element", "charge", "aromatic", "hcount"]
            )

            X.append(x)

        return X

    def _get_element_one_hot_index(self, atomic_number):
        # from of d4_50k_v0.2 train set, we have:
        #         Element Name  Atomic Number     Count
        # 0       Carbon              6  643122.0
        # 1     Nitrogen              7  112452.0
        # 2       Oxygen              8   98763.0
        # 3     Fluorine              9   15220.0
        # 4      Silicon             14       2.0
        # 5       Sulfur             16   12402.0
        # 6     Chlorine             17    3657.0
        # 7      Bromine             35    1043.0
        # 8       Iodine             53      64.0

        # map a known elt to corresponding index, else bin into 9
        if atomic_number in [6, 7, 8, 9]:
            return atomic_number - 6
        if atomic_number == 14:
            return 4
        if atomic_number in [16, 17]:
            return atomic_number - 11
        if atomic_number == 35:
            return 7
        if atomic_number == 53:
            return 8
        return 9  # else


# test dataset.py
if __name__ == "__main__":
    print("Processing dataset...")
    tr_loader, val_loader, test_loader = get_train_val_test_loaders(
        batch_size=get_config("model.batch_size")
    )
