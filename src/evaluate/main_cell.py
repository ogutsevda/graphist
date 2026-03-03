import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from utils import set_random_seed


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Cell-level logistic regression evaluation"
    )

    # Split mode
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["csv", "folder"],
        required=True,
        help="'csv': single emb dir + fold CSVs (NuCLS). "
        "'folder': separate fold directories (PanNuke).",
    )

    # csv mode args
    parser.add_argument(
        "--emb_dir",
        type=str,
        default=None,
        help="Directory with all .npz embeddings (csv mode)",
    )
    parser.add_argument(
        "--split_csv_dir",
        type=str,
        default=None,
        help="Directory with fold_X_train.csv / fold_X_test.csv (csv mode)",
    )

    # folder mode args
    parser.add_argument(
        "--fold_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Fold directories (folder mode), e.g. fold1/ fold2/ fold3/",
    )

    # Common args
    parser.add_argument(
        "--num_folds",
        type=int,
        default=None,
        help="Number of folds (auto-detected if not set)",
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--class_weight", type=str, default="balanced")

    return parser.parse_args()


# ---------- Loading helpers ----------


def load_cell_embeddings_from_dir(emb_dir):
    """Load all .npz files from a directory."""
    xs, ys = [], []
    all_npz_files = sorted(glob.glob(os.path.join(emb_dir, "*.npz")))

    for fp in tqdm(all_npz_files, desc=f"Loading {emb_dir}"):
        data = np.load(fp)
        emb = np.asarray(data["embedding"])
        lbl = np.asarray(data["labels"])

        if lbl.shape[0] != emb.shape[0]:
            raise ValueError(
                f"Label length {lbl.shape[0]} != num cells {emb.shape[0]} in {fp}"
            )

        xs.append(emb)
        ys.append(lbl)

    X = np.vstack(xs)
    y = np.concatenate(ys)
    return X, y


def load_cell_embeddings_by_slides(emb_dir, slide_ids):
    """Load .npz files whose filename starts with one of the given slide IDs."""
    xs, ys = [], []
    slide_id_set = set(slide_ids)
    all_npz_files = sorted(glob.glob(os.path.join(emb_dir, "*.npz")))

    files_to_load = []
    for fp in all_npz_files:
        # Slide ID is the first underscore-delimited token
        file_slide_id = os.path.basename(fp).split("_")[0]
        if file_slide_id in slide_id_set:
            files_to_load.append(fp)

    for fp in tqdm(sorted(files_to_load), desc="Loading embs"):
        data = np.load(fp)
        emb = np.asarray(data["embedding"])
        lbl = np.asarray(data["labels"])

        if lbl.shape[0] != emb.shape[0]:
            raise ValueError(
                f"Label length {lbl.shape[0]} != num cells {emb.shape[0]} in {fp}"
            )

        xs.append(emb)
        ys.append(lbl)

    X = np.vstack(xs)
    y = np.concatenate(ys)
    return X, y


# ---------- Train & evaluate ----------


def train_and_eval_cell_level(
    X_train,
    y_train,
    X_test,
    y_test,
    C=1.0,
    max_iter=2000,
    class_weight="balanced",
    random_state=0,
):

    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)

    # Pipeline: Standardize → Logistic Regression
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    solver="lbfgs",
                    multi_class="multinomial",
                    C=C,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)  # shape: [N, num_classes]

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)

    # AUROC & AUPRC
    if n_classes == 2:
        auroc = roc_auc_score(y_test, y_proba[:, 1])
        auprc = average_precision_score(y_test, y_proba[:, 1])
    else:
        auroc = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="macro",
        )
        auprc = average_precision_score(
            y_test,
            y_proba,
            average="macro",
        )

    # Print results
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Macro F1          : {f1_macro:.4f}")
    print(f"  Balanced Accuracy : {balanced_acc:.4f}")
    print(f"  AUROC (macro OvR) : {auroc:.4f}")
    print(f"  AUPRC (macro OvR) : {auprc:.4f}")

    cm_test = confusion_matrix(y_test, y_pred, labels=classes)
    print(f"  Confusion matrix:\n{cm_test}\n")

    result = {
        "num_train_cells": int(X_train.shape[0]),
        "num_test_cells": int(X_test.shape[0]),
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "balanced_accuracy": float(balanced_acc),
        "auroc_macro_ovr": float(auroc),
        "auprc_macro_ovr": float(auprc),
        "classification_report": report_dict,
    }

    return result


# ---------- Cross-validation runners ----------


def run_csv_mode(args):
    assert args.emb_dir is not None
    assert args.split_csv_dir is not None

    # Auto-detect number of folds
    if args.num_folds is None:
        fold_files = glob.glob(os.path.join(args.split_csv_dir, "fold_*_train.csv"))
        args.num_folds = len(fold_files)
    print(f"Running {args.num_folds}-fold CV (csv mode)\n")

    results = []
    for i in range(1, args.num_folds + 1):
        print(f"--- Fold {i}/{args.num_folds}: test on fold {i} ---")

        train_csv = os.path.join(args.split_csv_dir, f"fold_{i}_train.csv")
        test_csv = os.path.join(args.split_csv_dir, f"fold_{i}_test.csv")

        train_slides = pd.read_csv(train_csv)["slide_name"].tolist()
        test_slides = pd.read_csv(test_csv)["slide_name"].tolist()

        X_train, y_train = load_cell_embeddings_by_slides(args.emb_dir, train_slides)
        X_test, y_test = load_cell_embeddings_by_slides(args.emb_dir, test_slides)

        result = train_and_eval_cell_level(
            X_train,
            y_train,
            X_test,
            y_test,
            C=args.C,
            max_iter=args.max_iter,
            class_weight=args.class_weight,
            random_state=args.seed,
        )
        result["fold"] = i
        results.append(result)

    return results


def run_folder_mode(args):
    assert args.fold_dirs is not None and len(args.fold_dirs) >= 2
    num_folds = len(args.fold_dirs)
    print(f"Running {num_folds}-fold CV (folder mode)\n")

    results = []
    for i in range(num_folds):
        test_dir = args.fold_dirs[i]
        train_dirs = [d for j, d in enumerate(args.fold_dirs) if j != i]

        print(f"--- Fold {i+1}/{num_folds}: test on {test_dir} ---")

        # Load training data from all other folds
        X_tr_list, y_tr_list = [], []
        for d in train_dirs:
            X_f, y_f = load_cell_embeddings_from_dir(d)
            X_tr_list.append(X_f)
            y_tr_list.append(y_f)
        X_train = np.vstack(X_tr_list)
        y_train = np.concatenate(y_tr_list)

        X_test, y_test = load_cell_embeddings_from_dir(test_dir)

        result = train_and_eval_cell_level(
            X_train,
            y_train,
            X_test,
            y_test,
            C=args.C,
            max_iter=args.max_iter,
            class_weight=args.class_weight,
            random_state=args.seed,
        )
        result["test_fold"] = test_dir
        result["train_folds"] = train_dirs
        results.append(result)

    return results


# ---------- Main ----------


def main():
    args = parse_arguments()

    set_random_seed(args.seed)
    print(f"Seed: {args.seed}")

    if args.split_mode == "csv":
        results = run_csv_mode(args)
    else:
        results = run_folder_mode(args)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print summary
    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    print(f"\n===== SUMMARY =====")
    print(f"  Mean accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean macro F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Results saved to {results_path}")
    print(f"===================\n")


if __name__ == "__main__":
    main()
