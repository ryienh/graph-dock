"""
Train GCN
    Trains a graph neural network to predict docking score based on subset of docking data
    Periodically outputs training information, and saves model checkpoints
    Usage: python3 train.py

For questions or comments, contact rhosseini@anl.gov
"""

import torch
from dataset import get_train_val_test_loaders
from model import GINREG
import tqdm
import wandb
from scipy.stats import pearsonr

from utils import get_config, restore_checkpoint, save_checkpoint


def _train_epoch(data_loader, model, optimizer, device):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    model = model.train()
    running_loss = 0

    for X in tqdm.tqdm(data_loader):
        # handle cuda
        X = X.to(device)

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


def _evaluate_epoch(
    val_loader,
    model,
    stats,
    device,
    train_loss,
):

    model = model.eval()

    running_loss = 0
    with torch.no_grad():

        predictions = []
        labels = []

        for X in tqdm.tqdm(val_loader):
            X = X.to(device)

            prediction = model(X)
            prediction = torch.squeeze(prediction)
            loss = model.loss(prediction, X.y)

            # loss calculation
            running_loss += loss.item() * X.num_graphs

            predictions += prediction.tolist()
            labels += X.y.tolist()

        running_loss /= len(val_loader.dataset)

    stats.append([running_loss, train_loss])
    return running_loss, pearsonr(predictions, labels)


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
        ),
    )

    hyperparams = wandb.config

    # generate data or load from file
    print("Starting training...")

    tr_loader, va_loader, _ = get_train_val_test_loaders(
        batch_size=hyperparams["batch_size"]
    )

    # define model, loss function, and optimizer
    model = GINREG(
        input_dim=hyperparams["node_feature_size"],
        hidden_dim=hyperparams["hidden_dim"],
        dropout=hyperparams["dropout"],
        num_conv_layers=hyperparams["num_conv_layers"],
    )
    model = model.double()
    wandb.watch(model, log_freq=1000)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print("Using device: ", device)

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
        va_loader, model, stats, device, 0
    )  # training loss and accuracy for training is 0 first

    # Loop over the entire dataset multiple times
    best_val_loss = 100

    for epoch in range(start_epoch, hyperparams["num_epochs"]):
        # Train model
        train_loss = _train_epoch(tr_loader, model, optimizer, device)
        print(f"Train loss for epoch {epoch} is {train_loss}.")

        # Evaluate model
        val_loss, pearson_coef = _evaluate_epoch(
            va_loader, model, stats, device, train_loss
        )
        print(f"Val loss for epoch {epoch} is {val_loss}.")

        # update if best val loss
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = val_loss
            wandb.run.summary["best_val_loss_epoch"] = epoch
            wandb.run.summary["best_pearson_coef"] = pearson_coef

        # Call logger
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        # Save model parameters
        save_checkpoint(model, epoch + 1, get_config("model.checkpoint"), stats)

    print("Finished Training")


if __name__ == "__main__":
    main()
