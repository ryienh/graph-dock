"""
Run inference on trained model
    Runs inference on trained network on dataset of arbitrary length. Labels and predictions are 
    saved to file
    Usage: python3 inference.py

For questions or comments, contact rhosseini@anl.gov
"""
import torch
import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

from utils import get_config, restore_checkpoint
from dataset import get_train_val_test_loaders
from model import *
from utils import get_degree_hist


def _get_model(tr_dataset):

    model_name = get_config("model.name")

    if model_name == "GINREGv0.1":
        model = GINREG(
            input_dim=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            dropout=get_config("model.dropout"),
            num_conv_layers=get_config("model.num_conv_layers"),
        )

    elif model_name == "PNAREGv0.1":
        deg = get_degree_hist(tr_dataset.dataset)
        # deg.to(device)
        model = PNAREG(
            input_dim=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            dropout=get_config("model.dropout"),
            num_conv_layers=get_config("model.num_conv_layers"),
            deg=deg,
        )

    elif (
        model_name == "GATREGv0.1"
        or model_name == "GATREGv0.1small"
        or model_name == "GATREGv0.1med"
    ):
        model = GATREG(
            input_dim=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            dropout=get_config("model.dropout"),
            num_conv_layers=get_config("model.num_conv_layers"),
            heads=get_config("model.num_heads"),
        )

    elif model_name == "AttentiveFPREGv0.1":
        model = AttentiveFPREG(
            input_dim=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            dropout=get_config("model.dropout"),
            num_conv_layers=get_config("model.num_conv_layers"),
            num_out_channels=get_config("model.output_dim"),
            edge_dim=1,
            num_timesteps=get_config("model.num_timesteps"),
        )

    else:
        raise NotImplementedError(f"{model_name} not yet implemented.")

    return model


def _forward_inference(loader, model, device):

    model = model.eval()

    running_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():

        for X in tqdm.tqdm(loader):

            X = X.to(device)

            thresh, _ = torch.sort(X.y)
            thresh = thresh[int(X.y.shape[0] / 10)]  # FIXME: fix hardcode

            logits = model(X)

            prediction = torch.squeeze(logits)
            loss = model.loss(prediction, X.y, get_config("model.exp_weighing"))

            # loss calculation
            running_loss += loss.item() * X.num_graphs

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

    # get validation set
    tr_loader, va_loader, _ = get_train_val_test_loaders(
        batch_size=get_config("model.batch_size")
    )

    # define model, task
    model = _get_model(tr_loader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(torch.float64)
    model = model.to(device)

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
    ) = _forward_inference(va_loader, model, device)

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
    np.savetxt(
        f"./outputs/{name}_exp{weight}_labels.csv", labels, delimiter=",", fmt="%f"
    )
    np.savetxt(
        f"./outputs/{name}_exp{weight}_preds.csv", preds, delimiter=",", fmt="%f"
    )


if __name__ == "__main__":
    main()
