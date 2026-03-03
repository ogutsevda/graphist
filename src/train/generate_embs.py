import os
import glob
import time
import h5py
import json
import torch
import datetime
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import Counter, defaultdict
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.nn import global_mean_pool

from models import build_model
from utils import (
    seed_worker,
    set_random_seed,
    split_data,
    load_checkpoint,
    GraphDataset,
    NormalizeData,
    AddVirtualNode,
)


def get_args_parser():

    parser = argparse.ArgumentParser("GrapHist embedding generation", add_help=False)

    # Dataset and seed
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--scale",
        type=str,
        choices=["slide", "patch", "cell"],
        required=True,
    )

    # Paths
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--norm_json_path", type=str, required=True)
    parser.add_argument(
        "--metadata_csv_path",
        type=str,
        default=None,
        help="Required for slide/patch scale",
    )
    parser.add_argument(
        "--cell_graphs_path",
        type=str,
        default=None,
        help="Required for cell scale",
    )
    parser.add_argument("--output_path", type=str, required=True)

    # Model parameters
    parser.add_argument("--encoder", type=str, default="acm_gin")
    parser.add_argument("--decoder", type=str, default="acm_gin")

    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--replace_rate", type=float, default=0.1)
    parser.add_argument("--node_pooling", type=str, default="mean")
    parser.add_argument("--num_hidden", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_out_heads", type=int, default=1)
    parser.add_argument("--num_edge_features", type=int, default=1)
    parser.add_argument("--residual", type=bool, default=None)
    parser.add_argument("--attn_drop", type=float, default=0.1)
    parser.add_argument("--in_drop", type=float, default=0.2)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    parser.add_argument("--batchnorm", type=bool, default=False)
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=3)
    parser.add_argument("--concat_hidden", type=bool, default=True)

    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)

    return parser


def save_patient_h5(pid, embeddings_list, filenames_list, output_path):
    arr = np.stack(embeddings_list, axis=0)  # shape = (n_tiles, hidden_dim)
    patient_path = os.path.join(output_path, pid)
    os.makedirs(patient_path, exist_ok=True)
    h5_path = os.path.join(patient_path, "embeddings.h5")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "embeddings", data=arr, maxshape=(None, arr.shape[1]), chunks=True
        )
        for idx, fname in enumerate(filenames_list):
            f.attrs[f"filename_{idx}"] = fname
        f.attrs["count"] = arr.shape[0]


def generate_embeddings_slide(
    model,
    data_loader,
    output_path,
    device,
    patient_patch_counts: Counter,
):
    """
    Generate WSI- and RoI-level embeddings for patched datasets.
    """

    os.makedirs(output_path, exist_ok=True)
    model.eval()

    emb_buffer = defaultdict(list)
    name_buffer = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding batches"):
            batch = batch.to(device)
            h = model.embed(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            h = global_mean_pool(h, batch.batch)

            for i, pid in enumerate(batch.sample_id):

                patient_path = os.path.join(output_path, pid)
                if os.path.exists(patient_path):
                    continue

                emb_buffer[pid].append(h[i].cpu().numpy())
                name_buffer[pid].append(batch.graph_path[i])

                if len(emb_buffer[pid]) == patient_patch_counts[pid]:
                    save_patient_h5(pid, emb_buffer[pid], name_buffer[pid], output_path)
                    del emb_buffer[pid]
                    del name_buffer[pid]


def generate_embeddings_patch(
    model,
    data_loader,
    output_path,
    device,
):
    """
    Generate RoI-level embeddings for non-patched datasets.
    """
    os.makedirs(output_path, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding patches"):
            batch = batch.to(device)
            h = model.embed(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            h = global_mean_pool(h, batch.batch)

            # Save each patch's embedding individually
            for i, path in enumerate(batch.graph_path):
                patch_name = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(output_path, f"{patch_name}.npy")
                np.save(out_path, h[i].cpu().numpy())


def generate_embeddings_cell(
    model,
    data_loader,
    output_path,
    device,
):
    """
    Generate cell-level embeddings for non-patched datasets.
    """

    os.makedirs(output_path, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding cells"):
            batch = batch.to(device)
            h = model.embed(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            # Save node embeddings and node labels for this patch
            for i, path in enumerate(batch.graph_path):
                patch_name = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(output_path, f"{patch_name}.npz")

                start = int(batch.ptr[i].item())
                end = int(batch.ptr[i + 1].item())
                emb_i = h[start : end - 1]  # remove the virtual node

                embedding = emb_i.detach().cpu().numpy()
                labels = batch.labels[start : end - 1].detach().cpu().numpy()
                np.savez(out_path, embedding=embedding, labels=labels)


def main():

    args = get_args_parser().parse_args()

    print("\n===== GRAPHIST EMBEDDING GENERATION =====")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=========================================\n")

    # Device & seeding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    g = torch.Generator()
    g.manual_seed(args.seed)

    with open(args.norm_json_path) as f:
        scale_vals = json.load(f)

    transforms = Compose(
        [
            ToUndirected(),
            NormalizeData(scale_vals),
            AddVirtualNode(),
        ]
    )

    print(f"--- Embedding generation with seed {args.seed} ---")
    set_random_seed(args.seed)

    if args.scale == "cell":
        # Cell-level: no metadata needed, just glob all .pt files
        assert args.cell_graphs_path is not None
        all_paths = sorted(glob.glob(os.path.join(args.cell_graphs_path, "*.pt")))
        print(f"Found {len(all_paths)} graph files in {args.cell_graphs_path}")

        ds = GraphDataset(all_paths, transform=transforms)
        args.num_features = int(ds[0].num_features)

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        # Slide/patch: read metadata CSV with splits
        assert args.metadata_csv_path is not None
        data_df = pd.read_csv(args.metadata_csv_path)

        # Compute train/val/test masks
        train_mask, val_mask, test_mask = split_data(data_df)

        # Paths & counts for train+val
        df_trainval = data_df[train_mask | val_mask].sort_values(by="sample_id")
        train_paths = df_trainval["graph_path"].tolist()
        train_counts = Counter(df_trainval["sample_id"].tolist())

        # Paths & counts for test
        df_test = data_df[test_mask].sort_values(by="sample_id")
        test_paths = df_test["graph_path"].tolist()
        test_counts = Counter(df_test["sample_id"].tolist())

        # Build datasets
        train_ds = GraphDataset(train_paths, transform=transforms)
        test_ds = GraphDataset(test_paths, transform=transforms)
        args.num_features = int(train_ds[0].num_features)

        # DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

    # Build & load model
    model = build_model(args).to(device)
    model = load_checkpoint(args.model_path, model, optimizer=None, just_model=True)
    model.eval()

    # Generate & save
    start = time.time()
    print("Generating & saving embeddings…")

    if args.scale == "slide":
        generate_embeddings_slide(
            model,
            train_loader,
            args.output_path,
            device,
            train_counts,
        )
        generate_embeddings_slide(
            model,
            test_loader,
            args.output_path,
            device,
            test_counts,
        )

    elif args.scale == "patch":
        generate_embeddings_patch(
            model,
            train_loader,
            args.output_path,
            device,
        )
        generate_embeddings_patch(
            model,
            test_loader,
            args.output_path,
            device,
        )

    elif args.scale == "cell":
        generate_embeddings_cell(
            model,
            loader,
            args.output_path,
            device,
        )

    else:
        raise ValueError(
            f"Unknown scale: {args.scale}. Must be one of: slide, patch, cell"
        )

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))

    print("\n===== SUMMARY =====")
    print(f"Output directory : {args.output_path}")
    print(f"Dataset          : {args.dataset}")
    print(f"Total time       : {elapsed}")
    print("===================\n")


if __name__ == "__main__":
    main()
