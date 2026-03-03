import os
import fire
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from stardist.models import StarDist2D

from utils import read_image, CustomNormalizer

# Configure TensorFlow to allow memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} Physical GPUs found.")
    except RuntimeError as e:
        print(e)


def segment_cells(input_path, output_path, model):
    """
    Returns the coordinates and probabilities of cell centroids using StarDist.
    """
    # Parameters for the StarDist model for 20x images
    scale, nms_thresh, prob_thresh = (1, 0.01, 0.25)

    # List all patient folders
    patient_folders = sorted(
        [
            os.path.join(input_path, d)
            for d in os.listdir(input_path)
            if os.path.isdir(os.path.join(input_path, d))
        ]
    )
    print(f"Processing {len(patient_folders)} patient folders")

    for patient_folder in tqdm(patient_folders, desc="Processing patient folders"):
        patient_name = os.path.basename(patient_folder)
        patient_out_path = os.path.join(output_path, patient_name)
        os.makedirs(patient_out_path, exist_ok=True)

        # List all patch images for the current patient
        patch_files = [
            f
            for f in os.listdir(patient_folder)
            if os.path.isfile(os.path.join(patient_folder, f))
        ]
        print(f"Processing {len(patch_files)} patches for patient {patient_name}")

        for patch_file in tqdm(
            patch_files, desc=f"Processing patches for {patient_name}", leave=False
        ):
            img_path = os.path.join(patient_folder, patch_file)
            img_name = Path(patch_file).stem

            # Output file paths
            probs_path = os.path.join(patient_out_path, f"{img_name}_probs.csv")
            coords_path = os.path.join(patient_out_path, f"{img_name}_coords.npy")

            # Skip if results already exist
            if os.path.exists(probs_path) and os.path.exists(coords_path):
                continue

            try:
                # Read and normalize the image
                curr_image = read_image(img_path)
                mi, ma = np.percentile(curr_image, [0.2, 99.8])
                if mi == ma:
                    mi = ma - 1
                normalizer = CustomNormalizer(mi, ma)

                # Run the StarDist model
                labels, polys = model.predict_instances(
                    curr_image,
                    axes="YXC",
                    normalizer=normalizer,
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh,
                    scale=scale,
                    return_labels=False,
                    return_predict=False,
                    predict_kwargs=dict(verbose=False),
                )

                # Save probabilities and coordinates
                probabilities = list(polys["prob"])
                pd.DataFrame(probabilities, columns=["probability"]).to_csv(
                    probs_path, index=False
                )
                np.save(coords_path, polys["coord"])

            except Exception as e:
                print(f"Error processing image {img_name}: {e}")


def main(
    input_path: str,
    output_path: str,
):

    # Load the StarDist model
    model = StarDist2D.from_pretrained("2D_versatile_he")
    model.config.use_gpu = True

    segment_cells(input_path, output_path, model)


if __name__ == "__main__":
    fire.Fire(main)
