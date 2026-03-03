### Generate delaunay triangulation graphs and save them
import os
import glob
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import Delaunay
from joblib import Parallel, delayed


def delaunay_triangulation(
    cells_df,
    x_coords,
    y_coords,
    dist_threshold,
    x_coords_micron="center_X_micron",
    y_coords_micron="center_Y_micron",
):

    cells_coordinates = cells_df[[x_coords, y_coords]].to_numpy()
    delaunay_triplets = Delaunay(cells_coordinates)
    simp = delaunay_triplets.simplices

    edges = np.vstack([simp[:, [0, 1]], simp[:, [1, 2]], simp[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    i, j = edges[:, 0], edges[:, 1]

    coords_um = cells_df[[x_coords_micron, y_coords_micron]].to_numpy()
    xi, yi = coords_um[i, 0], coords_um[i, 1]
    xj, yj = coords_um[j, 0], coords_um[j, 1]

    dist = np.hypot(xi - xj, yi - yj)
    keep = dist <= dist_threshold

    delaunay_edges_df = pd.DataFrame(
        {
            "row_id_1": i[keep],
            "row_id_2": j[keep],
            "distance": dist[keep],
        }
    )

    return delaunay_edges_df


def construct_graph(
    features_path: str,
    output_path: str,
    dist_threshold: int,
):
    patient_name = os.path.basename(os.path.dirname(features_path))
    edges_filename = os.path.basename(features_path).replace("features_20x", "edges")

    output_dir = os.path.join(output_path, patient_name)
    os.makedirs(output_dir, exist_ok=True)

    edge_file = os.path.join(output_dir, edges_filename)

    if os.path.exists(edge_file):
        return

    if os.path.getsize(features_path) == 0:
        print(f"Skipping empty file: {features_path}")
        return

    try:
        region_detections = pd.read_csv(features_path)

        # Drop rows where any coordinate is NaN or infinite
        region_detections = region_detections[
            region_detections[["center_X", "center_Y"]].notnull().all(axis=1)
        ]
        region_detections = region_detections[
            np.isfinite(region_detections[["center_X", "center_Y"]]).all(axis=1)
        ]

        if len(region_detections) < 10:
            return

    except pd.errors.EmptyDataError:
        print(f"EmptyDataError: No data in file {features_path}")
        return

    except KeyError as e:
        print("Failed to read CSV file for", features_path)
        return

    region_detections["cell_id"] = region_detections.index
    region_detections.reset_index(drop=True, inplace=True)
    region_detections["row_id"] = region_detections.index

    try:
        delaunay_df = delaunay_triangulation(
            region_detections, "center_X", "center_Y", dist_threshold
        )

        delaunay_df.to_csv(edge_file, index=False)

    except KeyError as e:
        print("Failed to compute Delaunay triangulation for", features_path)
        return

    return


def main(
    feat_path,
    output_path,
    dist_threshold=100,
):
    os.makedirs(output_path, exist_ok=True)

    # Get all patient folders
    patient_folders = sorted(glob.glob(os.path.join(feat_path, "*/")))
    print(len(patient_folders), "patient folders found")

    # Collect all feature files from patient folders
    features_paths = []
    for patient_folder in patient_folders:
        cell_features = glob.glob(os.path.join(patient_folder, "*.csv"))
        features_paths.extend(cell_features)
        print(
            f"Found {len(cell_features)} features in {os.path.basename(os.path.dirname(patient_folder))}"
        )

    features_paths.sort()
    print(f"Processing a total of {len(features_paths)} feature files")

    Parallel(n_jobs=8)(
        delayed(construct_graph)(
            features_path,
            output_path,
            dist_threshold,
        )
        for features_path in tqdm(features_paths)
    )


if __name__ == "__main__":
    fire.Fire(main)
