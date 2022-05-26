"""
Run inference on trained model
    Runs inference on trained network on dataset of arbitrary length. Labels and predictions are
    saved to file
    Usage: python3 inference.py
"""

import random
import torch
import torch_geometric
import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
)

from utils import get_config, restore_checkpoint
from dataset import get_train_val_test_loaders
from model import *
from train import loss

from pyg_utils import VirtualNode


def _get_model():

    model_name = get_config("model.name")

    if model_name == "FiLMRegv0.1":
        model_ = FiLMReg(
            input_dim=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            dropout=get_config("model.dropout"),
            num_conv_layers=get_config("model.num_conv_layers"),
        )

    elif model_name == "FiLMRegv0.2":
        model_ = FiLMv2Reg(
            input_dim=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            dropout=get_config("model.dropout"),
            num_conv_layers=get_config("model.num_conv_layers"),
        )

    elif model_name == "GINREGv0.1":
        model_ = GINREG(
            input_dim=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            dropout=get_config("model.dropout"),
            num_conv_layers=get_config("model.num_conv_layers"),
        )

    elif model_name == "GATREGv0.1":
        model_ = GATREG(
            input_dim=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            dropout=get_config("model.dropout"),
            num_conv_layers=get_config("model.num_conv_layers"),
            heads=get_config("model.num_heads"),
        )

    else:
        raise NotImplementedError(f"{model_name} not yet implemented.")

    return model_


def _forward_inference(loader, model, device):

    model = model.eval()

    running_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():

        model = model.eval()

        with torch.no_grad():

            for X in tqdm.tqdm(loader):

                X = X.to(device)
                X.y = X.y.to(torch.float32)

                logits = model(X)

                prediction = torch.squeeze(logits)
                my_loss = loss(prediction, X.y, get_config("model.exp_weighing"))

                # loss calculation
                running_loss += my_loss.item() * X.num_graphs

                predictions += prediction.tolist()
                labels += X.y.tolist()

            running_loss /= len(loader.dataset)

    return (
        predictions,
        labels,
        running_loss,
        pearsonr(labels, predictions)[0],
        r2_score(labels, predictions),
        spearmanr(labels, predictions)[0],
        kendalltau(labels, predictions)[0],
        mean_absolute_error(labels, predictions),
    )


def main():

    print("Loading trained model...")

    torch.manual_seed(100)
    random.seed(100)
    np.random.seed(100)

    FULL_INF = get_config("full_inference")

    data_transform = torch_geometric.transforms.Compose([VirtualNode()])
    if FULL_INF is False:
        # get validation set
        _, va_loader, _ = get_train_val_test_loaders(
            batch_size=get_config("model.batch_size"), transform=data_transform
        )

    else:
        te_loader = get_train_val_test_loaders(
            batch_size=get_config("model.batch_size"),
            transform=data_transform,
            full_inf=True,
        )

    loader = te_loader if FULL_INF else va_loader

    # define model, task
    model = _get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Error, must use cuda")
        exit(1)
    model = model.to(torch.float32)
    model = model.to(device)  # single gpu for inf

    config_pth = get_config("model.checkpoint")
    pth = (
        f'checkpoints/{get_config("model.name")}_exp{get_config("model.exp_weighing")}'
        if config_pth.lower() == "auto"
        else config_pth
    )

    model, _ = restore_checkpoint(model, pth)
    print("Model successfully loaded")

    # forward inference on val set
    print("Beginning inference...")

    (
        preds,
        labels,
        val_loss,
        pearson_coef,
        r2,
        spearman,
        kendall,
        mae,
    ) = _forward_inference(loader, model, device)

    # print loss, other metrics
    print(f"Loss: {val_loss}.")
    print(f"Pearson coef: {pearson_coef}")
    print(f"R2: {r2}")
    print(f"Spearman: {spearman}")
    print(f"Kendall: {kendall}")
    print(f"MAE: {mae}")

    # save pred and labels to file
    name = get_config("model.name")
    weight = str(get_config("model.exp_weighing"))
    weight.replace(".", "-")

    if FULL_INF is False:
        np.savetxt(
            f"./outputs/{name}_exp{weight}_labels.csv",
            labels,
            delimiter=",",
            fmt="%f",
        )
        np.savetxt(
            f"./outputs/{name}_exp{weight}_preds.csv",
            preds,
            delimiter=",",
            fmt="%f",
        )

    else:
        np.savetxt(
            f"./outputs/{name}_exp{weight}_labels_FI_{chunk}.csv",
            labels,
            delimiter=",",
            fmt="%f",
        )
        np.savetxt(
            f"./outputs/{name}_exp{weight}_preds_FI_{chunk}.csv",
            preds,
            delimiter=",",
            fmt="%f",
        )


if __name__ == "__main__":
    main()
