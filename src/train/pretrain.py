import os
import json
import torch
import wandb
import argparse
import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from models import build_model
from utils import (
    seed_worker,
    set_random_seed,
    split_data,
    create_optimizer,
    load_checkpoint,
    save_checkpoint,
    GracefulKiller,
    GraphDataset,
    NormalizeData,
    AddVirtualNode,
)


def get_args():

    parser = argparse.ArgumentParser()

    # Dataset and seed
    parser.add_argument("--dataset", type=str, default="TCGA_BRCA")
    parser.add_argument("--seed", type=int, default=0)

    # Paths
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--norm_json_path", type=str, required=True)
    parser.add_argument("--metadata_csv_path", type=str, required=True)

    # Model parameters
    parser.add_argument("--encoder", type=str, default="acm_gin")
    parser.add_argument("--decoder", type=str, default="acm_gin")

    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--mask_rate", type=float, required=True)
    parser.add_argument("--replace_rate", type=float, required=True)
    parser.add_argument("--node_pooling", type=str, default="mean")
    parser.add_argument("--num_hidden", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_out_heads", type=int, default=1)
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
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler", type=bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--resume_run", type=bool, default=False)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--logging", type=bool, default=True)

    return parser.parse_args()


def pretrain(
    model,
    dataloaders,
    optimizer,
    max_epoch,
    device,
    scheduler,
    run_name,
    save_model=True,
    logs=True,
    save_folder=".",
    run_id=None,
    start_epoch=0,
    best_loss=1e8,
    node_pooler="mean",
    killer=None,
):

    train_loader, val_loader = dataloaders
    best_loss = best_loss

    for epoch in range(start_epoch, max_epoch):
        model.train()
        loss_list = []
        for batch_g in train_loader:
            if not killer.kill_now:
                batch_g = batch_g.to(device)

                model.train()
                loss = model(batch_g)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
            else:
                save_checkpoint(
                    os.path.join(save_folder, f"{run_name}_kill_checkpoint.pt"),
                    model,
                    optimizer,
                    epoch,
                    best_loss,
                    run_id,
                )
                exit(1)

        mean_train_loss = np.mean(loss_list)

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            model.eval()
            val_loss_list = []
            graph_embeddings = []
            for i, batch_g in enumerate(val_loader):
                batch_g = batch_g.to(device)
                loss = model(batch_g)
                val_loss_list.append(loss.item())

                if epoch % 10 == 0 and i % 2 == 0:
                    node_embedding = model.embed(
                        batch_g.x,
                        batch_g.edge_index,
                        batch_g.edge_attr,
                        batch_g.batch,
                    )

                    if node_pooler == "mean":
                        graph_embedding = (
                            global_mean_pool(node_embedding, batch_g.batch)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    elif node_pooler == "max":
                        graph_embedding = (
                            global_max_pool(node_embedding, batch_g.batch)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    elif node_pooler == "sum":
                        graph_embedding = (
                            global_add_pool(node_embedding, batch_g.batch)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    graph_embeddings.append(graph_embedding)

            mean_val_loss = np.mean(val_loss_list)

        if logs:
            wandb.log(
                {
                    "train_loss": mean_train_loss,
                    "val_loss": mean_val_loss,
                },
                step=epoch,
            )

        if graph_embeddings:
            # Choose a random 50% to conduct PCA on
            graph_embeddings = np.concatenate(graph_embeddings, axis=0)
            pca = IncrementalPCA(n_components=5, batch_size=40)
            pca.fit(graph_embeddings)
            exp_var_ratio = pca.explained_variance_ratio_
            print(exp_var_ratio)
            pca_dict = {
                "pca_1": exp_var_ratio[0],
                "pca_2": exp_var_ratio[1],
                "pca_3": exp_var_ratio[2],
                "pca_4": exp_var_ratio[3],
                "pca_5": exp_var_ratio[4],
            }
            if logs:
                wandb.log(pca_dict, step=epoch)

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            if save_model:
                save_checkpoint(
                    os.path.join(save_folder, f"{run_name}_checkpoint.pt"),
                    model,
                    optimizer,
                    epoch,
                    best_loss,
                    run_id,
                )

        print(
            f"Epoch {epoch} | train_loss: {mean_train_loss:.4f}, val_loss: {mean_val_loss:.4f}"
        )

    return model, optimizer, best_loss


def main(args):
    killer = GracefulKiller()

    dataset_name = args.dataset
    seed = args.seed

    encoder_type = args.encoder
    decoder_type = args.decoder

    drop_edge_rate = args.drop_edge_rate
    mask_rate = args.mask_rate
    replace_rate = args.replace_rate
    node_pooler = args.node_pooling
    num_hidden = args.num_hidden
    num_layers = args.num_layers

    optim_type = args.optimizer
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    use_scheduler = args.scheduler
    weight_decay = args.weight_decay

    train = args.train
    load_model = args.load_model
    resume_run = args.resume_run
    save_model = args.save_model
    logs = args.logging

    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    norm_json_path = args.norm_json_path
    metadata_csv_path = args.metadata_csv_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    g = torch.Generator()
    g.manual_seed(seed)

    data_df = pd.read_csv(metadata_csv_path)

    with open(norm_json_path) as json_file:
        scale_vals = json.load(json_file)

    transforms = Compose(
        [
            ToUndirected(),
            NormalizeData(scale_vals),
            AddVirtualNode(),
        ]
    )

    print(f"--- Run with seed {seed} ---")
    set_random_seed(seed)

    train_mask, val_mask, test_mask = split_data(data_df)

    train_dataset = GraphDataset(
        data_df[train_mask]["graph_path"].tolist(), transform=transforms
    )
    val_dataset = GraphDataset(
        data_df[val_mask]["graph_path"].tolist(), transform=transforms
    )
    args.num_features = int(train_dataset[0].num_features)
    args.num_edge_features = int(train_dataset[0].edge_attr.size(1))

    print("------------GrapHist---------")
    print(f"Train graphs: {len(train_dataset)}")
    print(f"Val graphs: {len(val_dataset)}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print("Train loader created")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print("Val loader created")

    run_name = "YOUR/RUN/NAME"
    checkpoint_name = "YOUR/CHECKPOINT/NAME"

    if os.path.isfile(os.path.join(save_folder, f"{run_name}_kill_checkpoint.pt")):
        checkpoint_name = f"{run_name}_kill_checkpoint.pt"
        load_model = True
        resume_run = True

    model = build_model(args)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    if use_scheduler:
        print("Use scheduler")
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    if load_model:
        print("Loading model")
        model, optimizer, start_epoch, best_loss, run_id = load_checkpoint(
            os.path.join(save_folder, checkpoint_name), model, optimizer
        )
        model.to(device)
        if logs:
            if resume_run:
                wandb.init(
                    project="GrapHist",
                    config=args,
                    name=run_name,
                    id=run_id,
                    resume="allow",
                    dir="YOUR/WORKING/DIR",
                )
            else:
                run_id = wandb.util.generate_id()
                args.run_id = run_id

                wandb.init(
                    project="GrapHist",
                    config=args,
                    name=run_name,
                    id=run_id,
                    dir="YOUR/WORKING/DIR",
                )

    else:
        start_epoch, best_loss, run_id = 0, 1e8, None
        if logs:
            run_id = wandb.util.generate_id()
            args.run_id = run_id

            wandb.init(
                project="GrapHist",
                config=args,
                name=run_name,
                id=run_id,
                dir="YOUR/WORKING/DIR",
            )

    if train:
        model, optimizer, best_loss = pretrain(
            model,
            (train_loader, val_loader),
            optimizer,
            max_epoch,
            device,
            scheduler,
            run_name,
            save_model=save_model,
            logs=logs,
            save_folder=save_folder,
            run_id=run_id,
            start_epoch=start_epoch,
            best_loss=best_loss,
            node_pooler=node_pooler,
            killer=killer,
        )

        if save_model:
            print("Saving model")
            save_checkpoint(
                os.path.join(save_folder, f"{run_name}_final.pt"),
                model,
                optimizer,
                max_epoch,
                best_loss,
                run_id,
            )


if __name__ == "__main__":
    args = get_args()
    main(args)
