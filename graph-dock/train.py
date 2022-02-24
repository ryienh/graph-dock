"""
Train multiple models
    Trains a graph neural network to predict docking score based on subset of docking data
    Periodically outputs training information, and saves model checkpoints
    Usage: python3 train.py

For questions or comments, contact rhosseini@anl.gov
"""

from enum import Enum
import torch
from dataset import get_train_val_test_loaders
from model import *
import tqdm
import wandb
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

from utils import (
    get_config,
    restore_checkpoint,
    save_checkpoint,
    get_degree_hist,
    calc_threshold,
)

def _train_epoch(data_loader, model, optimizer, device, exp_weighing):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    model = model.train()
    running_loss = 0

    for X in tqdm.tqdm(data_loader):

        X = X.to(device)  # FIXME: fix dataloading issue

        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        prediction = model(X)
        prediction = torch.squeeze(prediction)
        loss = model.loss(prediction, X.y, exp_weighing)
        loss.backward()
        optimizer.step()

        # calculate loss
        running_loss += loss.item() * X.num_graphs

    running_loss /= len(data_loader.dataset)

    return running_loss
    #


def _evaluate_epoch(val_loader, model, device, threshold, exp_weighing):

    model = model.eval()

    running_loss = 0
    predictions = []
    labels = []

    with torch.no_grad():

        for X in tqdm.tqdm(val_loader):

            X = X.to(device)

            logits = model(X)   
            prediction = torch.squeeze(logits)
            loss = model.loss(prediction, X.y, exp_weighing)

            # loss calculation
            running_loss += loss.item() * X.num_graphs

            predictions += prediction.tolist()
            labels += X.y.tolist()

        running_loss /= len(val_loader.dataset)

    # TODO check logic: may require two thresholds
    predictions = np.array(predictions)
    labels = np.array(labels)

    pseudo_preds = predictions < threshold.item()
    pseudo_labels = labels < threshold.item()

    return (
        running_loss,
        pearsonr(labels, predictions)[0],
        r2_score(labels, predictions),
        spearmanr(labels, predictions)[0],
        kendalltau(labels, predictions)[0],
        mean_absolute_error(labels, predictions),
        accuracy_score(pseudo_labels, pseudo_preds),
        balanced_accuracy_score(pseudo_labels, pseudo_preds),
        f1_score(pseudo_labels, pseudo_preds),
        recall_score(pseudo_labels, pseudo_preds),
        precision_score(pseudo_labels, pseudo_preds),
    )



def main():
    # init wandb logger
    wandb.init(
        project="graph-dock",
        config=dict(
            architecture=get_config("model.name"),
            threshold=get_config("model.threshold"),
            exp_weighing=get_config("model.exp_weighing"),
            learning_rate=get_config("model.learning_rate"),
            num_epochs=get_config("model.num_epochs"),
            batch_size=get_config("model.batch_size"),
            node_feature_size=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            num_conv_layers=get_config("model.num_conv_layers"),
            dropout=get_config("model.dropout"),
            dataset=get_config("dataset_id"),
            num_heads=get_config("model.num_heads"),
            num_timesteps=get_config("model.num_timesteps"),
            output_dim=get_config("model.output_dim"),
        ),
    )

    hyperparams = wandb.config

    # generate data or load from file
    print("Starting training...")

    tr_loader, va_loader, _ = get_train_val_test_loaders(
        batch_size=hyperparams["batch_size"]
    )

    # cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)

    # TODO: move model instantiation to diff file
    # define model, loss function, and optimizer
    model_name = hyperparams["architecture"]

    if model_name == "GINREGv0.1":
        model = GINREG(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
        )

    elif model_name == "PNAREGv0.1":
        deg = get_degree_hist(tr_loader.dataset)
        deg.to(device)
        model = PNAREG(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
            deg=deg,
        )

    elif model_name == "GATREGv0.1":
        model = GATREG(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
            heads=hyperparams["num_heads"],
        )

    elif model_name == "AttentiveFPREGv0.1":
        model = AttentiveFPREG(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
            num_out_channels=hyperparams["output_dim"],
            edge_dim=1,
            num_timesteps=hyperparams["num_timesteps"],
        )

    else:
        raise NotImplementedError(f"{model_name} not yet implemented.")

    model = model.to(torch.float64)
    model = model.to(device)
    wandb.watch(model, log_freq=1000)

    params = sum(p.numel() for p in model.parameters())
    print(f"Num parameters: {params}")
    wandb.run.summary["num_params"] = params

    # put entire loader onto device
    # tr_loader.dataset.data.to(device)
    # va_loader.dataset.data.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Attempts to restore the latest checkpoint if exists (only if running single experiment)
    if get_config("sweep") == 0:
        print("Loading checkpoint...")
        config_pth = get_config("model.checkpoint")
        pth = f'checkpoints/{get_config("model.name")}_exp{get_config("model.exp_weighing")}' if config_pth.lower() == "auto" else config_path
        if os.path.exists(pth):
            model, start_epoch = restore_checkpoint(model, pth)
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    # set threshold
    percentile = hyperparams["threshold"]
    tr_loader.dataset.data.to(device)  # TODO: do this earlier for both tr and va
    threshold = calc_threshold(percentile, tr_loader.dataset)
    print(f"Threshold with chosen percentile {percentile} is {threshold}")
    wandb.run.summary["threshold"] = threshold

    # exp weighing 
    exp_weighing = hyperparams["exp_weighing"]

    # Evaluate model
    _evaluate_epoch(
        va_loader, model, device, threshold, exp_weighing
    )  # training loss and accuracy for training is 0 first

    # Loop over the entire dataset multiple times
    best_val_loss = float("inf")

    for epoch in range(start_epoch, hyperparams["num_epochs"]):
        # Train model
        train_loss = _train_epoch(tr_loader, model, optimizer, device, exp_weighing)
        print(f"Train loss for epoch {epoch} is {train_loss}.")

        # Evaluate model
        (
            val_loss,
            pearson_coef,
            r2,
            spearman,
            kendall,
            mae,
            acc_score,
            balanced_acc_score,
            f1,
            recall,
            precision,
        ) = _evaluate_epoch(va_loader, model, device, threshold, exp_weighing)

        print(f"Val loss for epoch {epoch} is {val_loss}.")

        # update if best val loss
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = val_loss
            wandb.run.summary["best_val_loss_epoch"] = epoch

        # Call logger
        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "r2_score": r2,
                "spearman": spearman,
                "kendall": kendall,
                "pearson": pearson_coef,
                "MAE": mae,
                "acc_score": acc_score,
                "balanced_acc_score": balanced_acc_score,
                "f1": f1,
                "recall": recall,
                "precision": precision,
            }
        )

        # Save model parameters
        if get_config("sweep") == 0:
            config_pth = get_config("model.checkpoint")
            pth = f'checkpoints/{get_config("model.name")}_exp{get_config("model.exp_weighing")}' if config_pth.lower() == "auto" else config_path
            if not os.path.exists(pth):
                os.makedirs(pth, exist_ok=False)
            save_checkpoint(model, epoch + 1, pth)

    print("Finished Training")


if __name__ == "__main__":
    main()
