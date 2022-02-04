"""
Collection of utility functions for training and evaluation scripts.

For questions or comments, contact rhosseini@anl.gov
"""
import os
import torch
import itertools
from torch_geometric.utils import degree


def get_config(attr, fname=os.path.join("./", "graph-dock", "config.json")):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.

    Parameters
    ----------
    attr : str
        Size of train+val+test sample
    fname : os.path, optional
        Path to config file, default is "./config.json"

    Returns
    -------
    Requested attribute
    """
    if not hasattr(get_config, "config"):
        with open(fname) as f:
            get_config.config = eval(f.read())
    node = get_config.config
    for part in attr.split("."):
        node = node[part]
    return node


# From https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, epoch, checkpoint_dir):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)


def restore_checkpoint(model, checkpoint_dir, cuda=True, force=False, pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    files = [
        fn
        for fn in os.listdir(checkpoint_dir)
        if fn.startswith("epoch=") and fn.endswith(".checkpoint.pth.tar")
    ]

    if not files:
        print("No saved models found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in files:
            epoch = i
        else:
            break

    if not force:
        print(
            f"Select epoch: Choose in range [0, {epoch}].",
            "Entering 0 will train from scratch.",
        )
        print(">> ", end="")
        in_epoch = int(input())
        if in_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if in_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0
    else:
        print(f"Select epoch: Choose in range [1, {epoch}].")
        in_epoch = int(input())
        if in_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir, f"epoch={in_epoch}.checkpoint.pth.tar")

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, in_epoch


def clear_checkpoint(checkpoint_dir):
    fnames = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in fnames:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint removed")


def get_degree_hist(train_dataset):
    deg = torch.zeros(5, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


def calc_threshold(percentile, train_dataset):
    """
    Get absolute threshold for class weighing/classification from "Top X percent value"
        e.g. percentile of 0.1 means "select top 10 percent of train labels as positive"
    """
    if percentile is None:
        return None
    if percentile.lower() == "none":
        return None

    # FIXME: optimize
    labels = torch.ones(
        (0), dtype=torch.int32, device="cuda"
    )  # FIXME: fix cuda hardcode
    for data in train_dataset:
        labels.cat(data.y)

    thresh, _ = torch.sort(labels)
    thresh = thresh[int(labels.shape[0] / int(percentile * 100))]
    return thresh
