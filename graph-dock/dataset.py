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
from sklearn.preprocessing import normalize
from pysmiles import read_smiles
from mendeleev import element

import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from utils import config, suppress_stdout_stderr


def get_train_val_test_loaders(batch_size=config("model.batch_size")):
    tr, va, te = get_train_val_test_dataset()

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader


def get_train_val_test_dataset():
    root = config("data_dir")
    tr = ChemDataset(root, "train")
    va = ChemDataset(root, "val")
    te = ChemDataset(root, "test")

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
        self.data = pd.read_csv(config("clean_data_path"))

        data_list = self._load_data_mem()

        data, slices = self.collate(data_list)

        if self.partition == "train":
            torch.save((data, slices), self.processed_paths[0])
        elif self.partition == "val":
            torch.save((data, slices), self.processed_paths[1])
        elif self.partition == "test":
            torch.save((data, slices), self.processed_paths[2])

    def _load_data_mem(self):
        """
        Loads a single data partition from file to memory
        """

        # TODO: port to pygeo system of transform, pre_transform, pre_filter

        # select appropriate partition
        df = self.data[self.data.partition == self.partition]

        # store dictionary of chem names -> graphs #TODO: implement this for inference
        X = df["smiles"]
        y = df["dockscore"]
        X = X.to_numpy()
        y = y.to_numpy()

        # convert data to graphs
        X = self._smiles_2_graph(X)

        # drop NaN (prev fill NaN values with minimum)
        y = y[~np.isnan(y)]

        # normalize labels
        y = (y - np.mean(y)) / np.sqrt(np.var(y))

        assert len(X) == len(y)

        for graph, label in zip(X, y):
            graph.y = torch.FloatTensor(label)

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
                elt = np.zeros(
                    115
                )  # 115 possible atomic numbers, can possibly reduce this later
                atomic_num = element(x.nodes[idx]["element"]).atomic_number
                elt[atomic_num] = 1
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


# test dataset.py
if __name__ == "__main__":
    print("Processing dataset...")
    tr_loader, val_loader, test_loader = get_train_val_test_loaders()
