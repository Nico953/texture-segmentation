
import numpy as np
from scipy.ndimage import uniform_filter

from clustering import standard_kmeans, compute_pcc


def add_multiplicative_noise(
    image, noise_level, distribution="gaussian",
    clip=True, random_state=None
):
    """Ajoute un bruit multiplicatif a une image."""

    image = np.asarray(image, dtype=float)

    if image.ndim != 2 or image.size == 0:
        raise ValueError("image doit etre une image 2D non vide")
    if not np.isfinite(image).all():
        raise ValueError("image contient des valeurs invalides")
    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level doit etre positif ou nul")
    if distribution not in ("gaussian", "uniform"):
        raise ValueError("Distribution invalide")
    if not isinstance(clip, bool):
        raise ValueError("clip doit etre un booleen")

    rng = np.random.RandomState(random_state)

    if distribution == "gaussian":
        noise = rng.normal(0, noise_level, image.shape)
    else:
        noise = rng.uniform(-noise_level, noise_level, image.shape)

    noisy = image * (1 + noise)

    if clip:
        noisy = np.clip(noisy, image.min(), image.max())

    return noisy


def energy_feature(filtered_image, window_size=16):
    """Calcule la moyenne locale de la valeur absolue."""

    filtered_image = np.asarray(filtered_image, dtype=float)

    if filtered_image.ndim != 2 or filtered_image.size == 0:
        raise ValueError("filtered_image doit etre une image 2D non vide")
    if not np.isfinite(filtered_image).all():
        raise ValueError("filtered_image contient des valeurs invalides")
    if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
        raise ValueError("window_size doit etre un entier positif")

    return uniform_filter(
        np.abs(filtered_image),
        size=window_size,
        mode="reflect"
    )


def energy_feature_stack(filtered_images, window_size=16):
    """Calcule les cartes de caracteristiques energetiques."""

    filtered_images = np.asarray(filtered_images, dtype=float)

    if filtered_images.ndim != 3 or filtered_images.size == 0:
        raise ValueError("filtered_images doit etre un tableau 3D non vide")

    return np.stack([
        energy_feature(image, window_size)
        for image in filtered_images
    ])


def run_pipeline(
    image, n_clusters, feature_type="fd",
    region_size=16, smooth_window=17, random_state=0
):
    """Pipeline Gabor, caracteristiques, selection et K-means."""

    from gabor import build_filter_bank, apply_filter_bank
    from fractal import fractal_feature_stack
    from feature_selection import select_features

    if feature_type not in ("fd", "energy"):
        raise ValueError("feature_type doit etre fd ou energy")

    kernels = build_filter_bank()
    filtered, _ = apply_filter_bank(image, kernels)

    if feature_type == "fd":
        features = fractal_feature_stack(filtered)
    else:
        features = energy_feature_stack(filtered)

    selected, indices, _ = select_features(
        features, region_size=region_size
    )

    labels = standard_kmeans(
        selected,
        n_clusters=n_clusters,
        smooth_window=smooth_window,
        random_state=random_state
    )

    return labels, len(indices)


def _pcc_summary(values):
    """Calcule la moyenne et la dispersion des PCC."""

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": values
    }


def compare_noise_robustness(
    image, ground_truth, n_clusters,
    noise_levels=(0.0, 0.1, 0.2, 0.4),
    distribution="gaussian",
    n_repeats=5,
    base_random_state=0
):
    """Compare la robustesse FD et energie sous bruit multiplicatif."""

    results = {}

    for noise_level in noise_levels:
        for feature_type in ("fd", "energy"):

            pccs = []

            for rep in range(n_repeats):
                seed = base_random_state + rep

                if noise_level == 0:
                    noisy = image
                else:
                    noisy = add_multiplicative_noise(
                        image,
                        noise_level,
                        distribution=distribution,
                        random_state=seed
                    )

                labels, _ = run_pipeline(
                    noisy,
                    n_clusters,
                    feature_type=feature_type,
                    random_state=seed
                )

                pcc, _ = compute_pcc(
                    labels, ground_truth, n_clusters
                )

                pccs.append(pcc)

            results[(feature_type, noise_level)] = _pcc_summary(pccs)

    return results


def compare_noise_vs_kmeans_variability(
    image, ground_truth, n_clusters, noise_level,
    feature_type="fd", n_repeats=5,
    fixed_kmeans_seed=0, distribution="gaussian"
):
    """Compare la variabilite du bruit et celle du K-means."""

    results = {}

    for mode in ("bruit_varie", "kmeans_varie"):
        pccs = []

        for rep in range(n_repeats):

            noise_seed = (
                rep if mode == "bruit_varie"
                else fixed_kmeans_seed
            )

            kmeans_seed = (
                fixed_kmeans_seed if mode == "bruit_varie"
                else rep
            )

            noisy = add_multiplicative_noise(
                image,
                noise_level,
                distribution=distribution,
                random_state=noise_seed
            )

            labels, _ = run_pipeline(
                noisy,
                n_clusters,
                feature_type=feature_type,
                random_state=kmeans_seed
            )

            pcc, _ = compute_pcc(
                labels, ground_truth, n_clusters
            )

            pccs.append(pcc)

        results[mode] = _pcc_summary(pccs)

    return results


def generate_interpretation(results, noise_levels):
    """Compare les pertes de PCC entre FD et energie."""

    first = noise_levels[0]
    last = noise_levels[-1]

    fd_loss = (
        results[("fd", first)]["mean"]
        - results[("fd", last)]["mean"]
    )

    energy_loss = (
        results[("energy", first)]["mean"]
        - results[("energy", last)]["mean"]
    )

    lines = [
        f"Degradation FD : {fd_loss:+.1%}",
        f"Degradation energie : {energy_loss:+.1%}"
    ]

    if abs(fd_loss - energy_loss) < 0.03:
        lines.append("Les deux methodes ont une robustesse comparable.")
    elif fd_loss < energy_loss:
        lines.append("La FD est plus robuste sur cet intervalle.")
    else:
        lines.append("La caracteristique energie est plus robuste.")

    lines.append(
        "La comparaison porte sur le pipeline complet, "
        "avec une nouvelle selection des caracteristiques "
        "a chaque niveau de bruit."
    )

    return "\n".join(lines)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    rng = np.random.RandomState(0)

    # Mosaique de quatre textures
    size = 64
    image = np.zeros((128, 128))

    x = np.arange(size)

    image[:64, :64] = rng.rand(size, size)
    image[:64, 64:] = np.tile(
        0.5 + 0.5 * np.sin(2 * np.pi * x / 4),
        (size, 1)
    )
    image[64:, :64] = np.tile(
        (0.5 + 0.5 * np.sin(2 * np.pi * x / 24))[:, None],
        (1, size)
    )
    image[64:, 64:] = (
        np.indices((size, size)).sum(axis=0) % 16 < 8
    ).astype(float)

    ground_truth = np.zeros((128, 128), dtype=int)
    ground_truth[:64, 64:] = 1
    ground_truth[64:, :64] = 2
    ground_truth[64:, 64:] = 3

    noise_levels = (0.0, 0.1, 0.2, 0.4)

    results = compare_noise_robustness(
        image,
        ground_truth,
        n_clusters=4,
        noise_levels=noise_levels,
        n_repeats=5
    )

    print("Comparaison FD et energie :")

    for level in noise_levels:
        fd = results[("fd", level)]
        energy = results[("energy", level)]

        print(
            f"Bruit {level:.1f} : "
            f"FD={fd['mean']:.1%} +/- {fd['std']:.1%} | "
            f"Energie={energy['mean']:.1%} +/- {energy['std']:.1%}"
        )

    print("\n" + generate_interpretation(results, noise_levels))

    # Choisir le niveau de bruit avec la plus forte dispersion FD
    noise_level_to_probe = max(
        noise_levels,
        key=lambda n: results[("fd", n)]["std"]
    )

    print(f"\nNiveau de bruit selectionne : {noise_level_to_probe}")
    print(
        f"Ecart-type FD : "
        f"{results[('fd', noise_level_to_probe)]['std']:.1%}"
    )

    # Comparaison des sources de variabilite
    variability = compare_noise_vs_kmeans_variability(
        image,
        ground_truth,
        n_clusters=4,
        noise_level=noise_level_to_probe,
        n_repeats=5
    )

    for mode, values in variability.items():
        print(
            f"{mode} : "
            f"{values['mean']:.1%} +/- {values['std']:.1%}"
        )

    # Courbe de robustesse
    plt.figure(figsize=(8, 5))

    for feature_type, label in (
        ("fd", "Dimension fractale"),
        ("energy", "Energie")
    ):
        means = [
            results[(feature_type, n)]["mean"]
            for n in noise_levels
        ]
        stds = [
            results[(feature_type, n)]["std"]
            for n in noise_levels
        ]

        plt.errorbar(
            noise_levels,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=label
        )

    plt.xlabel("Niveau de bruit multiplicatif")
    plt.ylabel("PCC moyen")
    plt.title("Robustesse au bruit : FD vs Energie")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
