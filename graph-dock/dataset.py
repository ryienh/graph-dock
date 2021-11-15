"""
Implements data loading and pre-processing for chemical docking pre-training
    Currently, this only supports SMILES formatted data (such as the D4 dataset used in the experiment).
    The only front facing function is get_train_val_test_loaders(...), which should be called from the
    training script.

For questions or comments, contact rhosseini@anl.gov
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from pysmiles import read_smiles
from torch_geometric.utils.covert import from_networkx
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from utils import config


def get_train_val_test_loaders():
    tr, va, te = get_train_val_test_dataset()

    batch_size = config("model.batch_size")
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=4)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=4)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=4)

    return tr_loader, va_loader, te_loader


def get_train_val_test_dataset():
    tr = ChemDataset("train")
    va = ChemDataset("val")
    te = ChemDataset("test")

    return tr, va, te


class ChemDataset(Dataset):
    def __init__(self, partition):
        """
        Reads in necessary data from disk.
        """
        super().__init__()

        if partition not in ["train", "val", "test"]:
            raise ValueError("Partition {} does not exist".format(partition))

        np.random.seed(0)
        self.partition = partition
        self.data = pd.read_csv(config("clean_data_path"))
        self.load_type = config("data_load_type")

        if self.load_type in ["cpu", "filesys"]:
            raise ValueError("Load type {} not yet implemented".format(self.load_type))
        elif self.load_type != "gpu":
            raise ValueError("{} is not a valid load type.".format(self.load_type))
        else:
            self.X, self.y = self._load_data_gpu()

        scale = 0.8 if self.partition == "train" else 0.1  # TODO: fix hardcode

        self.length = int(len(self.data) * scale)

    def __len__(self):
        return self.length - 1

    def __getitem__(self, idx):

        # first attempt will be to fit data in GPU memory
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

    def _load_data_gpu(self):
        """
        Loads a single data partition from file
        """

        # select appropriate partition
        df = self.data[self.data.partition == self.partition]

        # store dictionary of chem names -> graphs #TODO: implement this for inference
        X = df["smiles"]
        y = df["dockscore"]
        X = X.to_numpy()
        y = y.to_numpy()

        # convert data to graphs
        X = self._smiles_2_graph(X)

        # fill NaN values with minimum
        min_score = min(y)
        idx = np.where(np.isnan(y))
        y[idx] = min_score

        # normalize scores
        y = normalize(y.reshape(-1, 1), axis=0)

        return X, y

    def _smiles_2_graph(self, smiles_list):
        """
        Converts list of smiles strings to Pytorch geometric graphs
        """
        # TODO: fix list
        X = []
        for molecule in smiles_list:
            x = from_networkx(read_smiles(molecule))
            X.append(x)


# test dataset.py
if __name__ == "__main__":
    print("Testing dataset.py...")
    get_train_val_test_loaders()
