import torch
import signal
import random
import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import BaseTransform


class GraphDataset(Dataset):
    def __init__(self, graph_paths: list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.graph_paths = graph_paths

    @property
    def processed_file_names(self):
        return self.graph_paths

    def len(self):
        return len(self.graph_paths)

    def get(self, idx):
        graph_path = self.graph_paths[idx]
        graph = torch.load(graph_path, weights_only=False)
        graph.graph_path = graph_path
        return graph


class NormalizeData(BaseTransform):
    def __init__(self, scale_dict: dict, attrs: List[str] = ["x", "edge_attr"]):
        tensor_dict = defaultdict(lambda: defaultdict())  # convert the data to tensors
        for key, value in scale_dict.items():
            tensor_dict[key]["mean"] = torch.tensor(value["mean"])
            tensor_dict[key]["std"] = torch.tensor(value["std"])

        self.tensor_dict = tensor_dict
        self.attrs = attrs

    def forward(self, data: Data) -> Data:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    mean = self.tensor_dict[key]["mean"]
                    std = self.tensor_dict[key]["std"]
                    value = (value - mean) / std
                    store[key] = value
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AddVirtualNode(BaseTransform):
    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        device = data.x.device if data.x is not None else "cpu"

        virtual_node_feat = torch.zeros((1, data.x.size(-1)), device=device)
        data.x = torch.cat([data.x, virtual_node_feat], dim=0)

        row = torch.arange(num_nodes, device=device)
        col = torch.full((num_nodes,), num_nodes, device=device)
        new_edges = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

        data.virtual_node_index = torch.tensor([num_nodes], device=device)

        if data.edge_attr is not None:
            # Average edge weight from TCGA BRCA
            avg_edge_weight = 22.5
            new_edge_attr = torch.full(
                (2 * num_nodes, data.edge_attr.size(-1)), avg_edge_weight, device=device
            )
            data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_data(data_df: pd.DataFrame):
    """Extract train/val/test masks from dataframe with 'split' column."""
    train_mask = data_df["split"] == "train"
    val_mask = data_df["split"] == "val"
    test_mask = data_df["split"] == "test"

    return train_mask, val_mask, test_mask


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def create_optimizer(
    opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None
):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = torch.optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return torch.optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def load_checkpoint(checkpoint_fpath, model, optimizer, just_model=False):
    checkpoint = torch.load(checkpoint_fpath, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if just_model:
        return model
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return (
            model,
            optimizer,
            checkpoint["epoch"],
            checkpoint["best_loss"],
            checkpoint["run_id"],
        )


def save_checkpoint(checkpoint_fpath, model, optimizer, epoch, best_loss, run_id):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "run_id": run_id,
        },
        checkpoint_fpath,
    )
    return
