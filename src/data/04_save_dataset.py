import os
import glob
import fire
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
import multiprocessing as mp

from torch_geometric.data import Dataset, Data


class GraphDataset(Dataset):
    def __init__(
        self,
        feat_list: list,
        edge_list: list,
        clinical_data: pd.DataFrame,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(None, transform, pre_transform)

        self.feat_list = feat_list
        self.edge_list = edge_list
        self.clinical_data = clinical_data

    @property
    def processed_file_names(self):
        return (self.feat_list, self.edge_list)

    def len(self):
        return len(self.processed_file_names[0])

    def get(self, idx):
        feat_path, edge_path = self.feat_list[idx], self.edge_list[idx]
        patient_id = os.path.basename(os.path.dirname(feat_path))
        patch_id = os.path.basename(feat_path).replace("_features_20x.csv", "")

        try:
            label = self.clinical_data.loc[patient_id, "label"]

            # Read CSV files with optimized settings
            feats = pd.read_csv(
                feat_path,
                usecols=lambda x: x
                not in [
                    "",
                    "Unnamed: 0",
                    "center_X",
                    "center_Y",
                    "center_X_micron",
                    "center_Y_micron",
                    "cell_id",
                    "row_id",
                ],
            )
            feats = torch.from_numpy(feats.to_numpy(dtype=np.float32))

            # Read only needed columns from edge file
            edges_df = pd.read_csv(
                edge_path, usecols=["row_id_1", "row_id_2", "distance"]
            )
            edges = torch.from_numpy(
                edges_df[["row_id_1", "row_id_2"]].to_numpy(dtype=np.float32)
            ).T.to(torch.int64)
            edge_features = torch.from_numpy(
                edges_df[["distance"]].to_numpy(dtype=np.float32)
            )

            graph = Data(
                x=feats,
                edge_index=edges,
                edge_attr=edge_features,
                sample_id=patient_id,
                patient_id=patient_id,
                patch_id=patch_id,
                label=label,
            )

            return graph

        except Exception as e:
            print(f"Error processing graph {idx}: {e}")
            return None


def save_dataset(idx, feat_path, edge_path, clinical_data, output_path):
    assert os.path.basename(feat_path).replace("features_20x", "") == os.path.basename(
        edge_path
    ).replace("edges", "")

    try:
        data = GraphDataset([feat_path], [edge_path], clinical_data)
        graph = data[0]

        if not graph:
            return None

        graph_name = f"{graph.patient_id}_{graph.patch_id}.pt"
        graph_path = os.path.join(output_path, graph_name)

        # Skip if already exists
        if os.path.exists(graph_path):
            return {
                "graph_path": graph_path,
                "sample_id": graph.sample_id,
                "label": graph.label,
            }

        # Save the graph
        torch.save(graph, graph_path)

        return {
            "graph_path": graph_path,
            "sample_id": graph.sample_id,
            "label": graph.label,
        }

    except Exception as e:
        print(f"Error processing {feat_path}: {e}")
        return None


def main(
    edge_path: str,
    clinical_data_path: str,
    output_path: str,
):
    os.makedirs(output_path, exist_ok=True)

    edge_folders = sorted(glob.glob(os.path.join(edge_path, "*/")))
    print(len(edge_folders), "patient folders found")

    edge_paths = []
    for edge_folder in edge_folders:
        edge_paths.extend(sorted(glob.glob(os.path.join(edge_folder, "*.csv"))))
    edge_paths.sort()

    # Assumes edge and feature folders are in the same directory
    # Also assumes the former has "graphs" in its name while the latter has "features"
    feat_folder = edge_folder.replace("graphs", "features")
    assert os.path.exists(feat_folder)

    feat_paths = [
        edge_path.replace("edges", "features_20x").replace("graphs", "features")
        for edge_path in edge_paths
    ]

    print(len(feat_paths), len(edge_paths))
    print(feat_paths[0], edge_paths[0])
    assert len(feat_paths) == len(edge_paths)

    # Load clinical data once
    print("Loading clinical data...")
    clinical_csv_path = os.path.join(clinical_data_path, "sample_labels.csv")
    clinical_data = pd.read_csv(clinical_csv_path).set_index("sample_id")

    # Determine number of processes
    num_processes = max(1, int(mp.cpu_count() * 0.75))
    print(f"Using {num_processes} processes for parallel processing")

    # Process graphs in parallel
    print(f"Processing {len(edge_paths)} graphs...")

    # Create a pool of workers
    with mp.Pool(processes=num_processes) as pool:
        # Create a partial function with fixed arguments
        process_func = partial(
            save_dataset,
            clinical_data=clinical_data,
            output_path=output_path,
        )

        # Apply the function to each item in parallel with a progress bar
        results = list(
            tqdm(
                pool.starmap(
                    process_func,
                    [(i, feat_paths[i], edge_paths[i]) for i in range(len(edge_paths))],
                ),
                total=len(edge_paths),
            )
        )

    # Filter out None results
    rows_list = [r for r in results if r is not None]

    print(f"Processed {len(rows_list)} graphs successfully")

    # Create and save the DataFrame
    df = pd.DataFrame(rows_list)

    # Load sample_split.csv and merge to add split column
    split_csv_path = os.path.join(clinical_data_path, "sample_split.csv")
    split_data = pd.read_csv(split_csv_path)

    df = df.merge(split_data, on="sample_id")

    csv_save_path = os.path.join(output_path, "metadata.csv")
    df.to_csv(csv_save_path, index=False)
    print(f"Results saved to {csv_save_path}")


if __name__ == "__main__":
    fire.Fire(main)
