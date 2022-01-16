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
from model import GINREG, PNAREG, PNACLF, GATREG
import tqdm
import wandb
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

from utils import get_config, restore_checkpoint, save_checkpoint, get_degree_hist


class Task(Enum):
    Clf = 1
    Reg = 2
    Ukn = 3


def _train_epoch(data_loader, model, optimizer, device):
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
        loss = model.loss(prediction, X.y)
        loss.backward()
        optimizer.step()

        # calculate loss
        running_loss += loss.item() * X.num_graphs

    running_loss /= len(data_loader.dataset)

    return running_loss
    #


def _evaluate_epoch(val_loader, model, stats, train_loss, device, task):

    model = model.eval()

    running_loss = 0
    predictions = []
    labels = []
    with torch.no_grad():

        for X in tqdm.tqdm(val_loader):

            X = X.to(device)

            logits = model(X)

            if task == Task.Reg:
                prediction = torch.squeeze(logits)
                loss = model.loss(prediction, X.y)
            elif task == Task.Clf:
                prediction = torch.argmax(logits, dim=1)
                logits = torch.squeeze(logits)
                loss = model.loss(logits, X.y)

            # loss calculation
            running_loss += loss.item() * X.num_graphs

            predictions += prediction.tolist()
            labels += X.y.tolist()

        running_loss /= len(val_loader.dataset)

    stats.append([running_loss, train_loss])

    if task == Task.Reg:
        return (
            running_loss,
            pearsonr(labels, predictions)[0],
            r2_score(labels, predictions),
            spearmanr(labels, predictions)[0],
            kendalltau(labels, predictions)[0],
            mean_absolute_error(labels, predictions),
        )
    elif task == Task.Clf:
        return (
            running_loss,
            accuracy_score(labels, predictions),
            balanced_accuracy_score(labels, predictions),
            f1_score(labels, predictions),
            recall_score(labels, predictions),
            precision_score(labels, predictions),
        )
    else:
        raise ValueError(f"Task must be one of Clf or Reg, not {task}")


def main():
    # init wandb logger
    wandb.init(
        project="graph-dock",
        config=dict(
            architecture=get_config("model.name"),
            learning_rate=get_config("model.learning_rate"),
            num_epochs=get_config("model.num_epochs"),
            batch_size=get_config("model.batch_size"),
            node_feature_size=get_config("model.node_feature_size"),
            hidden_dim=get_config("model.hidden_dim"),
            num_conv_layers=get_config("model.num_conv_layers"),
            dropout=get_config("model.dropout"),
            dataset=get_config("dataset_id"),
            threshold=get_config("threshold"),
            num_heads=get_config("model.num_heads"),
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

    # define model, loss function, task keyword, and optimizer
    model_name = hyperparams["architecture"]
    task = Task.Ukn

    if model_name == "GINREGv0.1":
        model = GINREG(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
        )
        task = Task.Reg

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
        task = Task.Reg

    elif model_name == "PNACLFv0.1":
        deg = get_degree_hist(tr_loader.dataset)
        deg.to(device)
        model = PNACLF(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
            deg=deg,
            threshold=hyperparams["threshold"],
        )
        task = Task.Clf

    elif model_name == "GATREGv0.1":
        print(f"Node Feature size: ", hyperparams["node_feature_size"])
        print(f"Node Feature size: ", hyperparams["node_feature_size"])
        model = GATREG(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
            heads=hyperparams["num_heads"],
        )
        task = Task.Reg

    else:
        raise NotImplementedError(f"{model_name} not yet implemented.")

    print(f"Task detected: {task}")

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
        model, start_epoch, stats = restore_checkpoint(
            model, get_config("model.checkpoint")
        )
    else:
        start_epoch = 0
        stats = []

    # Evaluate model
    _evaluate_epoch(
        va_loader, model, stats, 0, device, task
    )  # training loss and accuracy for training is 0 first

    # Loop over the entire dataset multiple times
    best_val_loss = 100

    for epoch in range(start_epoch, hyperparams["num_epochs"]):
        # Train model
        train_loss = _train_epoch(tr_loader, model, optimizer, device)
        print(f"Train loss for epoch {epoch} is {train_loss}.")

        # Evaluate model
        if task == Task.Reg:
            val_loss, pearson_coef, r2, spearman, kendall, mae = _evaluate_epoch(
                va_loader, model, stats, train_loss, device, task
            )
        elif task == Task.Clf:
            (
                val_loss,
                acc_score,
                balanced_acc_score,
                f1,
                recall,
                precision,
            ) = _evaluate_epoch(va_loader, model, stats, train_loss, device, task)
        else:
            raise ValueError(
                f'Invalid task, task must be one of "Clf" or "Reg" not {task}'
            )
        print(f"Val loss for epoch {epoch} is {val_loss}.")

        # update if best val loss
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = val_loss
            wandb.run.summary["best_val_loss_epoch"] = epoch

        # Call logger
        if task == Task.Reg:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "r2_score": r2,
                    "spearman": spearman,
                    "kendall": kendall,
                    "pearson": pearson_coef,
                    "MAE": mae,
                }
            )
        elif task == Task.Clf:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "acc_score": acc_score,
                    "balanced_acc_score": balanced_acc_score,
                    "f1": f1,
                    "recall": recall,
                    "precision": precision,
                }
            )

        # Save model parameters
        if get_config("sweep") == 0:
            save_checkpoint(model, epoch + 1, get_config("model.checkpoint"), stats)

    print("Finished Training")


if __name__ == "__main__":
    main()
