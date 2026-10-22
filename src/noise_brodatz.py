
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from brodatz import create_mosaic
from evaluate import add_multiplicative_noise, run_pipeline
from clustering import compute_pcc


def evaluate_robustness(
    image,
    ground_truth,
    noise_levels=(0.0, 0.1, 0.2, 0.4),
    n_repeats=5
):
    """Compare FD and energy under multiplicative noise."""

    results = {"fd": [], "energy": []}

    for noise_level in noise_levels:
        print(f"\nNoise level: {noise_level}")

        for feature_type in ("fd", "energy"):
            scores = []

            for repeat in range(n_repeats):
                noisy_image = add_multiplicative_noise(
                    image,
                    noise_level=noise_level,
                    distribution="gaussian",
                    random_state=repeat
                )

                labels, n_features = run_pipeline(
                    noisy_image,
                    n_clusters=4,
                    feature_type=feature_type,
                    region_size=16,
                    smooth_window=17,
                    random_state=repeat
                )

                pcc, _ = compute_pcc(
                    labels,
                    ground_truth,
                    n_clusters=4
                )

                scores.append(pcc * 100)

                print(
                    f"Feature={feature_type}, "
                    f"Repeat={repeat}, "
                    f"Selected={n_features}/12"
                )

            mean = np.mean(scores)
            std = np.std(scores)

            results[feature_type].append((mean, std))

            print(
                f"{feature_type}: "
                f"{mean:.2f}% +/- {std:.2f}%"
            )

    return results


def display_results(noise_levels, results):
    """Display PCC evolution under multiplicative noise."""

    plt.figure(figsize=(8, 5))

    for feature_type, label in (
        ("fd", "Fractal dimension"),
        ("energy", "Energy")
    ):
        means = [
            value[0] for value in results[feature_type]
        ]
        stds = [
            value[1] for value in results[feature_type]
        ]

        plt.errorbar(
            noise_levels,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=label
        )

    plt.xlabel("Multiplicative noise level")
    plt.ylabel("PCC (%)")
    plt.title("Noise robustness on Brodatz textures")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "brodatz"

    # Second Brodatz mosaic
    texture_paths = [
        data_dir / "D9.png",
        data_dir / "D68.png",
        data_dir / "D84.png",
        data_dir / "D112.png"
    ]

    mosaic, ground_truth = create_mosaic(texture_paths)

    noise_levels = (0.0, 0.1, 0.2, 0.4)

    results = evaluate_robustness(
        mosaic,
        ground_truth,
        noise_levels=noise_levels,
        n_repeats=5
    )

    display_results(noise_levels, results)
