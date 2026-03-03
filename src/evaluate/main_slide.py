import os
import json
import wandb
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

from mil.utils import DefaultMILGraph, DefaultAttentionModule
from utils import (
    EmbeddingsDataset,
    EarlyStopping,
    set_random_seed,
    seed_worker,
    get_classifier,
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # INPUT DATASET
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    # MODEL PARAMETERS
    parser.add_argument(
        "--mask_rate",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--replace_rate",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="attentive",
        choices=["attentive", "additive", "conjunctive"],
        help="Classifier type",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["ins-prob", None],
        help="Mode for the classifier",
    )
    # DATA PATHS
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
    )
    # WORKING AND SAVE DIRECTORIES
    parser.add_argument(
        "--project_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
    )
    # TRAINING DETAILS
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of splits for cross-validation"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Patience for early stopping"
    )
    parser.add_argument(
        "--delta", type=float, default=1e-2, help="Delta for early stopping"
    )

    return parser.parse_args()


def train(
    device,
    criterion,
    optimizer,
    train_loader,
    model,
):
    model.train()
    train_loss = 0.0
    train_true = []
    train_pred = []

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.x, torch.bincount(batch.batch))
        loss = criterion(preds["bag_logits"], batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_true.extend(batch.y.detach().cpu().numpy())
        train_pred.extend(
            torch.argmax(torch.softmax(preds["bag_logits"], dim=-1), dim=1)
            .detach()
            .cpu()
            .numpy()
        )

    mean_train_loss = train_loss / len(train_loader)
    train_f1 = f1_score(y_true=train_true, y_pred=train_pred, average="macro")

    return {"train_loss": mean_train_loss, "train_f1": train_f1}


def evaluate(
    device,
    criterion,
    val_loader,
    model,
):
    model.eval()
    val_loss = 0.0
    val_true = []
    val_pred = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch.x, torch.bincount(batch.batch))
            loss = criterion(preds["bag_logits"], batch.y)
            val_loss += loss.item()
            val_true.extend(batch.y.detach().cpu().numpy())
            val_pred.extend(
                torch.argmax(torch.softmax(preds["bag_logits"], dim=-1), dim=1)
                .detach()
                .cpu()
                .numpy()
            )

    mean_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(y_true=val_true, y_pred=val_pred, average="macro")

    return {"val_loss": mean_val_loss, "val_f1": val_f1}


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    set_random_seed(args.seed)
    print("Set random seed to", args.seed)

    # Create dataset and splits
    train_dataset = EmbeddingsDataset(
        args.train_data_path, args.label_path, args.input_dim
    )
    n_classes = train_dataset.num_labels
    args.output_dim = n_classes

    train_dataset_indices = list(range(len(train_dataset)))
    train_labels = [
        train_dataset.label_to_index[label] for label in train_dataset.label_strings
    ]

    test_dataset = EmbeddingsDataset(
        args.test_data_path, args.label_path, args.input_dim
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # Hyper parameter search
    attn_hidden_dim_list = [128, 256]
    clf_hidden_dim_list = [[], [args.input_dim]]

    dropout_list = [0.2, 0.5]
    lr_list = [0.001, 0.01]

    best_val_f1 = -float("inf")
    best_models = []
    saved_model_paths = []
    os.makedirs(args.save_dir, exist_ok=True)

    for clf_hidden_dim in clf_hidden_dim_list:
        for attn_hidden_dim in attn_hidden_dim_list:
            for dropout in dropout_list:
                for learning_rate in lr_list:
                    print(
                        f"Running experiment with clf_hidden_dim={clf_hidden_dim}, attn_hidden_dim={attn_hidden_dim}, dropout={dropout}, learning_rate={learning_rate}"
                    )

                    args.hidden_dim = clf_hidden_dim
                    args.hidden_dim_att = [attn_hidden_dim]
                    args.dropout = dropout
                    args.lr = learning_rate

                    fold_val_f1_list = []
                    fold_models = []

                    for fold, (train_fold_indices, val_fold_indices) in enumerate(
                        kf.split(train_dataset_indices, train_labels)
                    ):
                        print(f"Fold {fold+1}")

                        wandb.init(
                            project="slide-level-analysis",
                            config=args,
                            dir=args.project_dir,
                            mode="disabled",
                        )

                        train_dataset_fold = Subset(train_dataset, train_fold_indices)
                        val_dataset_fold = Subset(train_dataset, val_fold_indices)

                        train_labels_fold = [
                            train_labels[i] for i in train_fold_indices
                        ]
                        train_class_counts_fold = Counter(train_labels_fold)

                        class_counts = np.array(
                            [
                                train_class_counts_fold.get(i, 0)
                                for i in range(n_classes)
                            ]
                        )
                        class_weights = 1.0 / class_counts
                        sample_weights = np.array(
                            [class_weights[t] for t in train_labels_fold]
                        )
                        sample_weights = torch.from_numpy(sample_weights).double()
                        sampler = WeightedRandomSampler(
                            sample_weights,
                            len(sample_weights),
                            replacement=True,
                            generator=g,
                        )

                        classifier = get_classifier(args)
                        pointer = DefaultAttentionModule(
                            args.input_dim, args.hidden_dim_att
                        )
                        model = DefaultMILGraph(
                            pointer=pointer, classifier=classifier
                        ).to(device)

                        train_loader = DataLoader(
                            train_dataset_fold,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            worker_init_fn=seed_worker,
                            generator=g,
                        )
                        val_loader = DataLoader(
                            val_dataset_fold,
                            batch_size=args.batch_size,
                            shuffle=False,
                            worker_init_fn=seed_worker,
                            generator=g,
                        )

                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=args.lr)
                        scheduler = optim.lr_scheduler.StepLR(
                            optimizer, step_size=args.epochs / 10, gamma=0.5
                        )
                        early_stopping = EarlyStopping(
                            patience=args.patience, delta=args.delta
                        )

                        for epoch in tqdm(range(args.epochs)):
                            train_log_dict = train(
                                device=device,
                                criterion=criterion,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                model=model,
                            )

                            val_log_dict = evaluate(
                                device=device,
                                criterion=criterion,
                                val_loader=val_loader,
                                model=model,
                            )

                            scheduler.step()

                            log_dict = {**train_log_dict, **val_log_dict}
                            wandb.log(log_dict)

                            early_stopping(val_log_dict["val_loss"], model)
                            if early_stopping.early_stop:
                                print("Early stopping triggered")
                                break

                        if not early_stopping.early_stop:
                            print("Training completed without early stopping")
                        wandb.finish()

                        early_stopping.load_best_model(model)
                        fold_models.append(model)
                        best_val_log_dict = evaluate(
                            device=device,
                            criterion=criterion,
                            val_loader=val_loader,
                            model=model,
                            n_classes=n_classes,
                        )
                        fold_val_f1_list.append(best_val_log_dict["val_f1"])

                    mean_val_f1 = np.mean(fold_val_f1_list)
                    if mean_val_f1 > best_val_f1:
                        best_val_f1 = mean_val_f1
                        best_f1_config = {
                            "mean_val_f1": best_val_f1,
                            "attn_hidden_dim": attn_hidden_dim,
                            "clf_hidden_dim": clf_hidden_dim,
                            "dropout": dropout,
                            "learning_rate": learning_rate,
                        }
                        best_models = []
                        saved_model_paths = []
                        for fold_idx, fold_model in enumerate(fold_models, start=1):
                            model_path = os.path.join(
                                args.save_dir, f"best_model_fold{fold_idx}.pt"
                            )
                            torch.save(fold_model.state_dict(), model_path)
                            saved_model_paths.append(model_path)
                            best_models.append(fold_model)

    if not best_models:
        raise RuntimeError(
            "No best models were selected during hyperparameter search and cannot run testing."
        )

    test_metrics = {
        "test_loss": [],
        "test_f1": [],
    }

    for fold in range(args.n_splits):
        model = best_models[fold]

        test_log_dict = evaluate(
            device=device,
            criterion=criterion,
            val_loader=test_loader,
            model=model,
        )

        mean_test_loss = test_log_dict["val_loss"]
        test_f1 = test_log_dict["val_f1"]

        print(
            f"Fold {fold+1} Test Results: Loss: {mean_test_loss:.4f}, F1 Score: {test_f1:.4f}"
        )
        test_metrics["test_loss"].append(mean_test_loss)
        test_metrics["test_f1"].append(test_f1)

    test_loss_mean = np.mean(test_metrics["test_loss"])
    test_loss_std = np.std(test_metrics["test_loss"])
    test_f1_mean = np.mean(test_metrics["test_f1"])
    test_f1_std = np.std(test_metrics["test_f1"])

    print(
        f"Test Results: Loss: {test_loss_mean:.4f} ± {test_loss_std:.4f}, F1 Score: {test_f1_mean:.4f} ± {test_f1_std:.4f}"
    )

    results = {
        "test_loss": {"mean": test_loss_mean, "std": test_loss_std},
        "test_f1": {"mean": test_f1_mean, "std": test_f1_std},
        "best_val_config": best_f1_config,
        "saved_model_paths": saved_model_paths,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Test metrics saved to {results_path}")


if __name__ == "__main__":
    main()
