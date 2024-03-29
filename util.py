"""
Collection of auxiliary utility functions for graph-dock

"""
import pandas as pd
import random
import os


def preprocess_data(
    sample_size=50000,
    partition_ratios=[0.8, 0.1, 0.1],
    raw_data_path=os.path.join(
        "./", "data", "d4_table_name_smi_energy_hac_lte_25_title.csv"
    ),
    output_path=os.path.join("./", "data", "d4_dock_data_500k.csv"),
):
    """
    Formats data from docking data given by Lyu et al (classical docking) paper
    into format expected by training script.

    Parameters
    ----------
    sample_size : int
        Size of train+val+test sample
    partition_ratios : List[int]
        Ratios for training, validation, testing, respectively
    raw_data_path : os.path
        Location of raw data
    output_path : os.path
        Location of output data
    """
    n = sum(1 for line in open(raw_data_path)) - 1
    skip_r = sorted(random.sample(range(1, n + 1), n - sample_size))
    df = pd.read_csv(raw_data_path, skiprows=skip_r)
    df = df.drop(["hac"], axis=1)
    # create partitions
    partitions = (
        ["train"] * int(partition_ratios[0] * sample_size)
        + ["val"] * int(partition_ratios[1] * sample_size)
        + ["test"] * int(partition_ratios[2] * sample_size)
    )

    # deal with int rounding
    while len(partitions) != sample_size:
        partitions.append("train")

    random.shuffle(partitions)
    df["partition"] = partitions
    df.to_csv(output_path)


if __name__ == "__main__":
    preprocess_data()
