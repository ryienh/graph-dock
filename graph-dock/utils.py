"""
Collection of utility functions for training and evaluation scripts.

For questions or comments, contact rhosseini@anl.gov
"""
import os


def config(attr, fname=os.path.join("../", "config.json")):
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
    if not hasattr(config, "config"):
        with open(fname) as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split("."):
        node = node[part]
    return node
