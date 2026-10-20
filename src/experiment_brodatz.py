from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from brodatz import create_mosaic
from gabor import build_filter_bank, apply_filter_bank
from fractal import fractal_feature_stack
from feature_selection import select_features
from clustering import (
    standard_kmeans,
    iterative_kmeans,
    compute_pcc
)


def run_experiment():
    """Compare standard and iterative K-means on Brodatz textures."""

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "brodatz"

    texture_paths = [
        data_dir / "D9.png",
        data_dir / "D15.png",
        data_dir / "D29.png",
        data_dir / "D68.png"
    ]

    # Load the mosaic and ground truth
    mosaic, ground_truth = create_mosaic(texture_paths)

    # Apply the Gabor filter bank
    kernels = build_filter_bank()
    filtered, _ = apply_filter_bank(mosaic, kernels)

    # Compute fractal features
    features = fractal_feature_stack(filtered)

    # Select relevant features
    selected, indices, _ = select_features(
        features,
        region_size=16
    )

    print("Mosaic shape:", mosaic.shape)
    print("Ground truth shape:", ground_truth.shape)
    print("Selected features:", len(indices))
    print("Selected indices:", indices)

    # Standard K-means
    labels_standard = standard_kmeans(
        selected,
        n_clusters=4,
        smooth_window=17,
        random_state=0
    )

    pcc_standard, mapping_standard = compute_pcc(
        labels_standard,
        ground_truth,
        n_clusters=4
    )

    # Iterative K-means
    labels_iterative, status_history = iterative_kmeans(
        selected,
        n_clusters=4,
        window_sizes=(33, 25, 17),
        random_state=0
    )

    pcc_iterative, mapping_iterative = compute_pcc(
        labels_iterative,
        ground_truth,
        n_clusters=4
    )

    print(f"Standard K-means PCC: {pcc_standard * 100:.2f}%")
    print(f"Iterative K-means PCC: {pcc_iterative * 100:.2f}%")

    print("Standard mapping:", mapping_standard)
    print("Iterative mapping:", mapping_iterative)

    for window_size, mask in zip((33, 25, 17), status_history):
        print(
            f"Window {window_size}: "
            f"{np.count_nonzero(mask)} ambiguous pixels"
        )

    # Align predicted labels with ground truth classes
    aligned_standard = np.vectorize(
        mapping_standard.get
    )(labels_standard)

    aligned_iterative = np.vectorize(
        mapping_iterative.get
    )(labels_iterative)

    # Display the results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(
        mosaic,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    axes[0].set_title("Brodatz mosaic")

    axes[1].imshow(
        ground_truth,
        cmap="tab10",
        vmin=0,
        vmax=3,
        interpolation="nearest"
    )
    axes[1].set_title("Ground truth")

    axes[2].imshow(
        aligned_standard,
        cmap="tab10",
        vmin=0,
        vmax=3,
        interpolation="nearest"
    )
    axes[2].set_title(
        f"Standard K-means\nPCC: {pcc_standard * 100:.2f}%"
    )

    axes[3].imshow(
        aligned_iterative,
        cmap="tab10",
        vmin=0,
        vmax=3,
        interpolation="nearest"
    )
    axes[3].set_title(
        f"Iterative K-means\nPCC: {pcc_iterative * 100:.2f}%"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()