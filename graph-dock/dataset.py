"""
Implements data loading and pre-processing for chemical docking pre-training
    Currently, this only supports SMILES formatted data (such as the D4 dataset used in the experiment).
    The only front facing function is get_train_val_test_loaders(...), which should be called from the
    training script.

"""

import numpy as np
import pandas as pd
import tqdm
import os
from pysmiles import read_smiles
from mendeleev import element

import torch
import torch_geometric
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from temp import from_smiles, VirtualNode


from utils import get_config, suppress_stdout_stderr


def get_train_val_test_loaders(batch_size, transform=None, full_inf=False):

    if full_inf is False:

        tr, va, te = get_train_val_test_dataset(transform)

        tr_loader = DataLoader(
            tr, batch_size=batch_size, shuffle=True, pin_memory=False
        )
        va_loader = DataLoader(
            va, batch_size=batch_size, shuffle=False, pin_memory=False
        )
        te_loader = DataLoader(
            te, batch_size=batch_size, shuffle=False, pin_memory=False
        )

        return tr_loader, va_loader, te_loader

    else:
        te = get_train_val_test_dataset(transform, full_inf=True)
        te_loader = DataLoader(
            te, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        return te_loader


def get_train_val_test_dataset(transform=None, full_inf=False):

    root = os.path.join(get_config("data_dir"), get_config("dataset_id"))

    if full_inf is False:
        tr = ChemDataset(root, "train", transform=transform)
        va = ChemDataset(root, "val", transform=transform)
        te = ChemDataset(root, "test", transform=transform)

        return tr, va, te

    else:
        te = ChemDataset(root, "test", transform=transform)
        return te


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

        NOTE: This is the version of preprocessing used in our final results. All other are experimental.
        """

        # select appropriate partition
        df = self.data[self.data.partition == self.partition]

        # store dictionary of chem names -> graphs
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
        Converts list of smiles strings to pyg graph
        """
        X = []
        for molecule in tqdm.tqdm(smiles_list):
            X.append(from_smiles(molecule, with_hydrogen=False, kekulize=False))

        return X


# test dataset.py
if __name__ == "__main__":
    print("Processing dataset...")

    data_transform = torch_geometric.transforms.Compose([VirtualNode()])

    tr_loader, val_loader, test_loader = get_train_val_test_loaders(
        batch_size=get_config("model.batch_size"),
        transform=data_transform,
        full_inf=True,
    )
