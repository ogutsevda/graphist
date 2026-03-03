import os
import copy
import h5py
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Dataset, Data

from mil.attentive_mil import AttentiveClassifier
from mil.additive_mil import AdditiveClassifier
from mil.conjunctive_mil import ConjunctiveClassifier


classifier_dict = {
    "attentive": AttentiveClassifier,
    "additive": AdditiveClassifier,
    "conjunctive": ConjunctiveClassifier,
}


class EmbeddingsDataset(Dataset):
    def __init__(self, data_path, label_path, input_dim):
        """
        Initializes the dataset with embeddings and labels.
        """
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.input_dim = input_dim

        self.embeddings, self.labels = self.create_dataset(
            self.data_path, self.label_path, self.input_dim
        )

        if "TCGA" in self.data_path:
            valid_classes = [
                "Infiltrating duct carcinoma, NOS",
                "Lobular carcinoma, NOS",
            ]
            valid_indices = [
                i for i, label in enumerate(self.labels) if label in valid_classes
            ]
            self.embeddings = [self.embeddings[i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]

        # Build label mapping
        self.unique_labels = sorted(set(self.labels))
        self.label_to_index = {
            label: idx for idx, label in enumerate(self.unique_labels)
        }
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        # Convert labels to indices
        self.label_strings = self.labels.copy()
        self.labels = torch.tensor(
            [self.label_to_index[label] for label in self.labels]
        )
        self.num_labels = len(self.unique_labels)

        self.normalizer_values_path = os.path.join(
            self.data_path, "../train/normalization.json"
        )
        self.mean_vals = None
        self.std_vals = None

        self._load_or_compute_normalizer_values()

    def _load_or_compute_normalizer_values(self):
        if os.path.exists(self.normalizer_values_path):
            try:
                with open(self.normalizer_values_path, "r") as f:
                    values = json.load(f)
                    self.mean_vals = torch.tensor(values["mean"])
                    self.std_vals = torch.tensor(values["std"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading normalizer_values.json")
        else:
            self._compute_normalizer_values()

    def _compute_normalizer_values(self):
        sum_embeddings = torch.zeros(self.input_dim)
        sum_sq_embeddings = torch.zeros(self.input_dim)
        total_instances = 0

        for embedding in tqdm(self.embeddings, desc="Computing normalizer values"):
            embedding = torch.tensor(embedding, dtype=torch.float32)
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)

            if embedding.shape[1] != self.input_dim:
                raise ValueError
            sum_embeddings += torch.sum(embedding, dim=0).to(torch.float64)
            sum_sq_embeddings += torch.sum(embedding**2, dim=0).to(torch.float64)
            total_instances += embedding.shape[0]

        if total_instances == 0:
            raise ValueError

        mean_vals = sum_embeddings / total_instances
        variance_vals = (sum_sq_embeddings / total_instances) - (mean_vals**2)
        std_vals = torch.sqrt(torch.clamp(variance_vals, min=1e-8))

        self.mean_vals = mean_vals.to(torch.float32)
        self.std_vals = std_vals.to(torch.float32)

        os.makedirs(os.path.dirname(self.normalizer_values_path), exist_ok=True)
        with open(self.normalizer_values_path, "w") as f:
            json.dump(
                {"mean": self.mean_vals.tolist(), "std": self.std_vals.tolist()},
                f,
                indent=4,
            )

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        label_idx = self.labels[idx]
        sample = Data(
            x=torch.FloatTensor(self.embeddings[idx]), y=label_idx.clone().detach()
        )
        return sample

    @staticmethod
    def create_dataset(data_path, label_path, input_dim):
        # Map each patient to their label
        label_df = pd.read_csv(label_path, sep=",", header=0)
        label_df = label_df[
            [
                "sample_id",
                "label",
            ]
        ]
        label_df = label_df.set_index("sample_id")

        # Create a dictionary with the patient id as key and the label as value
        sample_id_label = {}
        for index, row in label_df.iterrows():
            sample_id_label[index] = row["label"]

        labels = []
        embeddings = []
        sample_embedding_paths = list(os.listdir(data_path))
        for sample_id in sample_embedding_paths:
            try:
                file_path = os.path.join(data_path, sample_id, "embeddings.h5")
                with h5py.File(file_path, "r") as f:
                    sample_embeddings = np.asarray(f["embeddings"][:])

                    if sample_embeddings.shape[1] != input_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch. Expected {input_dim}, got {sample_embeddings.shape[1]}"
                        )

                class_label = sample_id_label.get(sample_id)

                if not class_label:
                    continue
                assert class_label is not None
                labels.append(class_label)
                embeddings.append(sample_embeddings)
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                continue

        return embeddings, labels


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


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


def get_classifier(args):
    classifier_input_dim = args.input_dim
    if args.classifier in ["additive", "conjunctive"]:
        return classifier_dict[args.classifier](
            input_dim=classifier_input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            mode=args.mode,
        )
    elif args.classifier == "attentive":
        return classifier_dict["attentive"](
            input_dim=classifier_input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown classifier: {args.classifier}")
