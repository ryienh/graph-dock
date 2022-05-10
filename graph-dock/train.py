"""
Train multiple models
    Trains a graph neural network to predict docking score based on subset of docking data
    Periodically outputs training information, and saves model checkpoints
    Usage: python3 train.py

For questions or comments, contact rhosseini@anl.gov
"""

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from dataset import get_train_val_test_loaders, get_train_val_test_dataset
from model import *
import tqdm
import wandb
import random
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
)

from utils import (
    get_config,
    restore_checkpoint,
    save_checkpoint,
    get_degree_hist,
    calc_threshold,
)

# multiprocess imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from temp import VirtualNode


# temp loss function for ddp FIXME
def loss(pred, label, exp_weighing=0):

    # vanilla mse loss if no coef given
    if exp_weighing == 0:
        return torch.nn.functional.mse_loss(pred, label)

    # else calculate unreduced (per datapoint) mse loss, calc weights, return mean
    out = torch.nn.functional.mse_loss(pred, label, reduction="none")
    weights = torch.exp(-1 * exp_weighing * label)
    return (weights * out).mean()


def setup(rank, world_size):
    # cleanup()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _train_epoch(data_loader, model, optimizer, rank, exp_weighing):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    model = model.train()
    running_loss = 0

    for X in tqdm.tqdm(data_loader):

        X.y = X.y.to(torch.float32)
        X = X.to(rank)  # FIXME: fix dataloading issue

        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        prediction = model(X)
        prediction = torch.squeeze(prediction)
        # loss = model.loss(prediction, X.y, exp_weighing)
        my_loss = loss(prediction, X.y, exp_weighing)
        my_loss.backward()
        optimizer.step()

        # calculate loss
        running_loss += my_loss.item() * X.num_graphs

    running_loss /= len(data_loader.dataset)

    return running_loss
    #


def _evaluate_epoch(val_loader, model, rank, threshold, exp_weighing):

    model = model.eval()

    running_loss = 0
    predictions = []
    labels = []

    with torch.no_grad():

        for X in tqdm.tqdm(val_loader):
            X = X.to(rank)
            X.y = X.y.to(torch.float32)

            logits = model(X)
            prediction = torch.squeeze(logits)
            my_loss = loss(prediction, X.y, exp_weighing)

            # loss calculation
            running_loss += my_loss.item() * X.num_graphs

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


def main(rank, world_size):

    # seed everything -- warning cuDNN, dataloaders, scat/gath ops not seeded
    torch.manual_seed(100)
    random.seed(100)
    np.random.seed(100)
    # torch.use_deterministic_algorithms(True)

    # init wandb logger
    if rank == 0 or rank is None:
        print("In rank 0")
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

    else:
        hyperparams = dict(
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
        )

    # generate data or load from file
    print("Starting training...")

    if rank is not None:
        # create default process group
        setup(rank, world_size)

    data_transform = torch_geometric.transforms.Compose([VirtualNode()])

    if rank == 0:
        _, va_loader, _ = get_train_val_test_loaders(
            batch_size=hyperparams["batch_size"], transform=data_transform
        )

    if rank is not None:
        train_dataset, _, _ = get_train_val_test_dataset(transform=data_transform)
        tr_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        tr_loader = DataLoader(
            train_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=False,  # bc we are using sampler
            pin_memory=False,
            sampler=tr_sampler,
        )
    else:  # rank is None
        tr_loader, va_loader, _ = get_train_val_test_loaders(
            batch_size=hyperparams["batch_size"], transform=data_transform
        )

    # cuda
    if rank is None:
        rank = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device: ", rank)

    # TODO: move model instantiation to diff file
    # define model, loss function, and optimizer
    model_name = hyperparams["architecture"]

    if model_name == "NovelRegv0.1":
        model_ = NovelReg(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
            heads=hyperparams["num_heads"],
        )

    elif model_name == "FiLMRegv0.1":
        model_ = FiLMReg(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
        )

    elif model_name == "GINREGv0.1":
        model_ = GINREG(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
        )

    elif model_name == "PNAREGv0.1":
        deg = get_degree_hist(tr_loader.dataset)
        deg.to(rank)
        model_ = PNAREG(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
            deg=deg,
        )

    elif (
        model_name == "GATREGv0.1"
        or model_name == "GATREGv0.1small"
        or model_name == "GATREGv0.1med"
    ):
        model_ = GATREG(
            input_dim=hyperparams["node_feature_size"],
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"],
            num_conv_layers=hyperparams["num_conv_layers"],
            heads=hyperparams["num_heads"],
        )

    elif model_name == "AttentiveFPREGv0.1":
        model_ = AttentiveFPREG(
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

    model_ = model_.to(torch.float32)
    model_ = model_.to(rank)

    if world_size is not None:
        model = DDP(model_, device_ids=[rank], find_unused_parameters=True)
    else:
        model = model_

    if rank == 0 or rank is None:
        wandb.watch(model, log_freq=1000)

    params = sum(p.numel() for p in model.parameters())
    print(f"Num parameters: {params}")
    if rank == 0 or rank is None:
        wandb.run.summary["num_params"] = params

    # put entire loader onto device
    # tr_loader.dataset.data.to(device)
    # va_loader.dataset.data.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Attempts to restore the latest checkpoint if exists (only if running single experiment)
    if get_config("sweep") == 0 and rank is None:
        print("Loading checkpoint...")
        config_pth = get_config("model.checkpoint")
        pth = (
            f'checkpoints/{get_config("model.name")}_exp{get_config("model.exp_weighing")}'
            if config_pth.lower() == "auto"
            else config_pth
        )
        if os.path.exists(pth):
            model, start_epoch = restore_checkpoint(model, pth)
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    # set threshold
    percentile = hyperparams["threshold"]
    tr_loader.dataset.data.to(rank)  # TODO: do this earlier for both tr and va
    threshold = calc_threshold(percentile, tr_loader.dataset)
    print(f"Threshold with chosen percentile {percentile} is {threshold}")
    if rank == 0 or rank is None:
        wandb.run.summary["threshold"] = threshold

    # exp weighing
    exp_weighing = hyperparams["exp_weighing"]

    # Evaluate model
    # if rank == 0 or rank is None:
    #     _evaluate_epoch(
    #         va_loader, model, rank, threshold, exp_weighing
    #     )  # training loss and accuracy for training is 0 first

    # Loop over the entire dataset multiple times
    best_val_loss = float("inf")

    for epoch in range(start_epoch, hyperparams["num_epochs"]):
        # Train model
        train_loss = _train_epoch(tr_loader, model, optimizer, rank, exp_weighing)
        if rank is not None:
            train_loss *= world_size
        print(f"Train loss for epoch {epoch} is {train_loss}.")
        dist.barrier()

        if (rank == 0 or rank is None) and (epoch % 10 == 0):  # get val every 10 epocs
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
            ) = _evaluate_epoch(va_loader, model, rank, threshold, exp_weighing)

            print(f"Val loss for epoch {epoch} is {val_loss}.")

            # update if best val loss
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                if rank == 0 or rank is None:
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
                pth = (
                    f'checkpoints/{get_config("model.name")}_exp{get_config("model.exp_weighing")}'
                    if config_pth.lower() == "auto"
                    else config_pth
                )
                if not os.path.exists(pth):
                    os.makedirs(pth, exist_ok=False)
                save_checkpoint(model, epoch + 1, pth)

        dist.barrier()

    cleanup()
    wandb.finish()
    print("Finished Training")


if __name__ == "__main__":
    RUN_WITH_MP = True

    if RUN_WITH_MP:
        WANDB_START_METHOD = "thread"
        world_size = 8
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        main(None, None)
