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


def run_experiment(texture_names, name):
    """Run both K-means methods on a Brodatz mosaic."""

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "brodatz"

    texture_paths = [
        data_dir / texture_name
        for texture_name in texture_names
    ]

    mosaic, ground_truth = create_mosaic(texture_paths)

    kernels = build_filter_bank()
    filtered, _ = apply_filter_bank(mosaic, kernels)

    features = fractal_feature_stack(filtered)

    selected, indices, _ = select_features(
        features,
        region_size=16
    )

    labels_standard = standard_kmeans(
        selected,
        n_clusters=4,
        smooth_window=17,
        random_state=0
    )

    labels_iterative, history = iterative_kmeans(
        selected,
        n_clusters=4,
        window_sizes=(33, 25, 17),
        random_state=0
    )

    pcc_standard, mapping_standard = compute_pcc(
        labels_standard,
        ground_truth,
        n_clusters=4
    )

    pcc_iterative, mapping_iterative = compute_pcc(
        labels_iterative,
        ground_truth,
        n_clusters=4
    )

    aligned_standard = np.array([
        mapping_standard[int(label)]
        for label in labels_standard.ravel()
    ]).reshape(mosaic.shape)

    aligned_iterative = np.array([
        mapping_iterative[int(label)]
        for label in labels_iterative.ravel()
    ]).reshape(mosaic.shape)

    print(f"\nMosaic {name}")
    print("Textures:", texture_names)
    print("Selected features:", len(indices))
    print(f"Standard PCC: {pcc_standard * 100:.2f}%")
    print(f"Iterative PCC: {pcc_iterative * 100:.2f}%")
    print(f"Difference: {(pcc_iterative - pcc_standard) * 100:+.2f} points")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(mosaic, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Mosaic {name}")

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
    axes[2].set_title(f"Standard: {pcc_standard * 100:.2f}%")

    axes[3].imshow(
        aligned_iterative,
        cmap="tab10",
        vmin=0,
        vmax=3,
        interpolation="nearest"
    )
    axes[3].set_title(f"Iterative: {pcc_iterative * 100:.2f}%")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return pcc_standard, pcc_iterative, len(indices)


if __name__ == "__main__":

    mosaic = {
    "A": ["D9.png", "D68.png", "D84.png", "D112.png"],
}

    results = {}

    for name, textures in mosaic.items():
        results[name] = run_experiment(textures, name)

    print("\nRESULTS SUMMARY")
    print("-" * 55)
    print("Mosaic | Standard | Iterative | Difference | Features")

    for name, (standard, iterative, n_features) in results.items():
        difference = (iterative - standard) * 100

        print(
            f"{name:^6} | "
            f"{standard * 100:7.2f}% | "
            f"{iterative * 100:8.2f}% | "
            f"{difference:+8.2f} | "
            f"{n_features:^8}"
        )