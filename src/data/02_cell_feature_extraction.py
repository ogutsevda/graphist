import os
import glob
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fft import rfft
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops_table
from utils import read_image, coords_to_roi


def centroid_distance_signature(X: np.ndarray, img_pixel_resolution: float):
    """
    Computes a signature that represents distance of polygon edges from centroid.
    """
    X = X * img_pixel_resolution
    means_ = X.mean(0)
    signature = (X[:, 0] - means_[0]) ** 2 + (X[:, 1] - means_[1]) ** 2
    signature = np.sqrt(signature)
    return signature


def add_feature_to_dict(
    measurements_dict: dict,
    features: list,
    feature_names: list,
    img_pixel_resolution: float,
) -> dict:

    if len(features) != len(feature_names):
        raise ValueError
    for curr_feature, curr_feature_name in zip(features, feature_names):
        if curr_feature_name in ["axis_major_length", "axis_minor_length", "perimeter"]:
            curr_feature = curr_feature * img_pixel_resolution
        elif curr_feature_name in ["area"]:
            curr_feature = curr_feature * (img_pixel_resolution**2)
        if np.ndim(curr_feature) == 1:
            if hasattr(curr_feature, "iloc"):
                measurements_dict[curr_feature_name] = curr_feature.iloc[0]
            else:
                measurements_dict[curr_feature_name] = curr_feature[0]
        else:
            measurements_dict[curr_feature_name] = curr_feature
    return measurements_dict


def get_cell_features(
    img,
    seg,
    probs,
    seg_coords,
    img_mpp,
    numpy_x,
    numpy_y,
    cell_id,
):

    where_seg_mask0 = np.where(seg == cell_id)
    [center_x, center_y] = np.median(where_seg_mask0, axis=1)
    center_x += numpy_x
    center_y += numpy_y
    min_coords = np.min(where_seg_mask0, axis=1)
    max_coords = np.max(where_seg_mask0, axis=1)

    curr_img = img[
        min_coords[0] : max_coords[0] + 1, min_coords[1] : max_coords[1] + 1, :
    ]
    curr_seg_mask = (
        1
        * (seg == cell_id)[
            min_coords[0] : min_coords[0] + curr_img.shape[0],
            min_coords[1] : min_coords[1] + curr_img.shape[1],
        ]
    )

    # Return if background
    if cell_id == 0:
        return

    measurements_dict = {}
    list_of_features = []
    features = []

    features.extend([center_x, center_y, center_x * img_mpp, center_y * img_mpp])
    list_of_features.extend(
        ["center_X", "center_Y", "center_X_micron", "center_Y_micron"]
    )

    masked_img = np.expand_dims(curr_seg_mask, axis=2) * curr_img
    where_seg_mask = np.where(curr_seg_mask == 1)
    masked_pixels = masked_img[where_seg_mask]

    # Compute color intensity features
    min_intensity = np.min(masked_pixels, axis=0)
    max_intensity = np.max(masked_pixels, axis=0)
    mean_intensity = np.mean(masked_pixels, axis=0)
    std_intensity = np.std(masked_pixels, axis=0)
    skew_intensity = skew(masked_pixels, axis=0)
    kurtosis_intensity = kurtosis(masked_pixels, axis=0)

    features.extend(
        list(min_intensity)
        + list(max_intensity)
        + list(mean_intensity)
        + list(std_intensity)
        + list(skew_intensity)
        + list(kurtosis_intensity)
    )
    list_of_features.extend(
        [
            "min_intensity_R",
            "min_intensity_G",
            "min_intensity_B",
            "max_intensity_R",
            "max_intensity_G",
            "max_intensity_B",
            "mean_intensity_R",
            "mean_intensity_G",
            "mean_intensity_B",
            "std_intensity_R",
            "std_intensity_G",
            "std_intensity_B",
            "skew_intensity_R",
            "skew_intensity_G",
            "skew_intensity_B",
            "kurtosis_intensity_R",
            "kurtosis_intensity_G",
            "kurtosis_intensity_B",
        ]
    )

    measurements_dict = add_feature_to_dict(
        measurements_dict, [probs.loc[cell_id - 1]], ["probability"], img_mpp
    )
    measurements_dict = add_feature_to_dict(
        measurements_dict, features, list_of_features, img_mpp
    )

    # Compute morphology features
    props = regionprops_table(
        curr_seg_mask,
        properties=(
            "orientation",
            "axis_major_length",  # µm
            "axis_minor_length",  # µm
            "eccentricity",  # ratio
            "area",  # µm^2
            "perimeter",  # µm
            "solidity",
            "extent",
        ),
    )

    circularities = [
        (4 * np.pi * area / (perimeter**2)) if perimeter != 0 else 0
        for area, perimeter in zip(props["area"], props["perimeter"])
    ]

    elongations = [
        (major / minor) if minor != 0 else 0
        for major, minor in zip(props["axis_major_length"], props["axis_minor_length"])
    ]

    props["circularity"] = np.array(circularities)
    props["elongation"] = np.array(elongations)

    measurements_dict = add_feature_to_dict(
        measurements_dict, props.values(), props.keys(), img_mpp
    )

    # Compute translation invariant Fourier features
    centered_polygon = (
        seg_coords[cell_id - 1].T - seg_coords[cell_id - 1].T.mean(axis=0)[None, :]
    )
    polygon_signature = centroid_distance_signature(centered_polygon, img_mpp)
    fourier_dict = {}

    # 20 and 30 are validated hyperparams from prior work
    for n_features in [20, 30]:
        fourier_features = np.abs(rfft(polygon_signature, n=n_features, workers=3))
        # Remove offset
        processed_fourier_features = fourier_features[1:].copy() / fourier_features[0]
        for i, x in enumerate(processed_fourier_features):
            fourier_dict["FD_%s_%s" % (n_features, i)] = x

    measurements_dict.update(fourier_dict)

    # Compute texture features
    grayco_dict = {}
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    gray_img = (rgb2gray(curr_img) * 255).astype(np.uint8)

    # Compute the gray-level co-occurrence matrix (GLCM)
    glcm = graycomatrix(gray_img, [1], angles, 256, symmetric=True, normed=True)

    properties = [
        "ASM",
        "contrast",
        "correlation",
        "dissimilarity",
        "energy",
        "homogeneity",
    ]

    for prop in properties:
        grayco_dict[f"mean_{prop}"] = np.mean(graycoprops(glcm, prop)[0])
        grayco_dict[f"std_{prop}"] = np.std(graycoprops(glcm, prop)[0])
        grayco_dict[f"skew_{prop}"] = skew(graycoprops(glcm, prop)[0])
        grayco_dict[f"kurtosis_{prop}"] = kurtosis(graycoprops(glcm, prop)[0])
        grayco_dict[f"min_{prop}"] = np.min(graycoprops(glcm, prop)[0])
        grayco_dict[f"max_{prop}"] = np.max(graycoprops(glcm, prop)[0])

    measurements_dict.update(grayco_dict)

    gray_scale_dict = {}
    gray_scale_dict["mean_intensity_gray_scale"] = np.mean(gray_img[where_seg_mask])
    gray_scale_dict["std_intensity_gray_scale"] = np.std(gray_img[where_seg_mask])
    gray_scale_dict["skew_intensity_gray_scale"] = skew(gray_img[where_seg_mask])
    gray_scale_dict["kurtosis_intensity_gray_scale"] = kurtosis(
        gray_img[where_seg_mask]
    )
    gray_scale_dict["min_intensity_gray_scale"] = np.min(gray_img[where_seg_mask])
    gray_scale_dict["max_intensity_gray_scale"] = np.max(gray_img[where_seg_mask])

    measurements_dict.update(gray_scale_dict)

    return measurements_dict


def compute_cell_features(
    coords_path: str,
    probs_path: str,
    input_path: str,
    output_path: str,
    patient_name: str,
    n_jobs=-1,
) -> None:
    """
    Returns cell level descriptors of morphology, texture, and intensity in csv format.
    """

    img_name = os.path.basename(coords_path).split("_coords")[0]
    img_folder = os.path.join(input_path, patient_name)
    img_path = os.path.join(img_folder, img_name + ".png")

    output_dir = os.path.join(output_path, patient_name)
    os.makedirs(output_dir, exist_ok=True)

    feature_file = os.path.join(output_dir, img_name + "_features_20x.csv")
    if os.path.exists(feature_file):
        return

    img = read_image(img_path)

    # Variables for 20x images
    img_mpp = 0.50
    img_dims = img.shape[0:2]
    resize_factor = 1

    seg_coords = np.load(coords_path) / resize_factor
    seg = coords_to_roi(seg_coords, img_dims)
    probs = pd.read_csv(probs_path)

    numpy_x, numpy_y = 0, 0

    def _process(cell_id):
        return get_cell_features(
            img, seg, probs, seg_coords, img_mpp, numpy_x, numpy_y, cell_id
        )

    all_measurements = Parallel(n_jobs=n_jobs)(
        delayed(_process)(d) for d in np.unique(seg)
    )

    all_measurements = [
        measurement for measurement in all_measurements if measurement is not None
    ]

    if all_measurements:
        all_measurements_df = pd.DataFrame(all_measurements)
        all_measurements_df.to_csv(feature_file)
    else:
        print(f"No valid detections found for {img_name}")
        return


def main(
    input_path: str,
    seg_path: str,
    output_path: str,
    n_jobs=24,
):

    # Get all patient folders within the segmentation directory
    patient_folders = sorted(glob.glob(os.path.join(seg_path, "*")))

    pbar = tqdm(patient_folders, desc="Processing patients")

    for patient_folder in pbar:
        patient_name = os.path.basename(patient_folder)
        pbar.set_description(f"Processing {patient_name}")

        coords_paths = sorted(glob.glob(os.path.join(patient_folder, "*_coords.npy")))

        for coords_path in coords_paths:
            probs_path = coords_path.replace("_coords.npy", "_probs.csv")

            compute_cell_features(
                coords_path,
                probs_path,
                input_path,
                output_path,
                patient_name,
                n_jobs=n_jobs,
            )


if __name__ == "__main__":
    fire.Fire(main)
