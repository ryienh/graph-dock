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

from utils import config, restore_checkpoint, save_checkpoint


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
        loss = model.loss(prediction, X.y)
        loss.backward()
        optimizer.step()

        # calculate loss
        running_loss = loss.item() * X.num_graphs

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

    model.eval()

    running_loss = 0
    with torch.no_grad():

        for X in tqdm.tqdm(val_loader):
            X = X.to(device)

            prediction = model(X)
            loss = model.loss(prediction, X.y)

            # loss calculation
            running_loss = loss.item() * X.num_graphs

        running_loss /= len(val_loader.dataset)

    stats.append([running_loss, train_loss])
    return running_loss


def main():
    # init wandb logger
    wandb.init(
        project="graph-dock",
        config={
            "learning_rate": config("model.learning_rate"),
            "architecture": config("model.name"),
            "batch_size": config("model.batch_size"),
            "hidden_dim": config("model.hidden_dim"),
            "dataset": config("dataset_id"),
        },
    )

    # generate data or load from file
    print("Starting training...")

    tr_loader, va_loader, _ = get_train_val_test_loaders()

    # define model, loss function, and optimizer
    model = GINREG(
        input_dim=config("model.node_feature_size"),
        hidden_dim=config("model.hidden_dim"),
        dropout=config("model.dropout"),
        num_conv_layers=config("model.num_conv_layers"),
    )
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config("model.learning_rate"))

    # cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Using device: ", device)

    # Attempts to restore the latest checkpoint if exists
    print("Loading checkpoint...")
    model, start_epoch, stats = restore_checkpoint(model, config("model.checkpoint"))

    # Evaluate model
    _evaluate_epoch(
        va_loader, model, stats, device, 0
    )  # training loss and accuracy for training is 0 first

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config("model.num_epochs")):
        # Train model
        train_loss = _train_epoch(tr_loader, model, optimizer, device)
        print(f"Train loss for epoch {epoch} is {train_loss}.")

        # Evaluate model
        val_loss = _evaluate_epoch(va_loader, model, stats, device, train_loss)
        print(f"Val loss for epoch {epoch} is {val_loss}.")

        # Call logger
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("model.checkpoint"), stats)

    print("Finished Training")


if __name__ == "__main__":
    main()
