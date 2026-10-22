
import numpy as np


def f_statistic(feature_map, region_size=16):
    """Calcule le score F pour une carte de caracteristiques."""

    feature_map = np.asarray(feature_map, dtype=float)

    if feature_map.ndim != 2 or not np.isfinite(feature_map).all():
        raise ValueError("feature_map doit etre une image 2D valide")

    if not isinstance(region_size, (int, np.integer)) or region_size <= 0:
        raise ValueError("region_size doit etre un entier positif")

    h, w = feature_map.shape
    rows = h // region_size
    cols = w // region_size
    k = rows * cols

    if k < 2:
        raise ValueError("Au moins deux regions sont necessaires")

    # Conserver uniquement les regions completes
    cropped = feature_map[:rows * region_size, :cols * region_size]

    # Organiser les pixels en blocs de meme taille
    blocks = cropped.reshape(
        rows, region_size, cols, region_size
    ).transpose(0, 2, 1, 3)

    means = blocks.mean(axis=(2, 3), keepdims=True)

    rss_t = np.sum((blocks - means) ** 2)
    rss_o = np.sum((cropped - cropped.mean()) ** 2)

    if rss_t == 0:
        return 0.0 if rss_o == 0 else float("inf")

    n = cropped.size

    return ((n - k) * max(0.0, rss_o - rss_t)) / ((k - 1) * rss_t)


def select_features(feature_stack, region_size=16, threshold_ratio=4.5):
    """Selectionne les caracteristiques selon le seuil Fmax / 4.5."""

    feature_stack = np.asarray(feature_stack, dtype=float)

    if feature_stack.ndim != 3 or feature_stack.shape[0] == 0:
        raise ValueError("feature_stack doit etre un tableau 3D non vide")

    if not np.isfinite(feature_stack).all():
        raise ValueError("feature_stack contient des valeurs invalides")

    if not np.isfinite(threshold_ratio) or threshold_ratio <= 1:
        raise ValueError("threshold_ratio doit etre superieur a 1")

    f_values = np.array([
        f_statistic(feature, region_size)
        for feature in feature_stack
    ])

    f_max = f_values.max()

    if f_max <= 0:
        raise ValueError("Aucune caracteristique informative")

    if np.isinf(f_max):
        selected_indices = np.flatnonzero(np.isinf(f_values)).tolist()
    else:
        threshold = f_max / threshold_ratio
        selected_indices = np.flatnonzero(f_values >= threshold).tolist()

    selected_stack = feature_stack[selected_indices]

    return selected_stack, selected_indices, f_values


if __name__ == "__main__":

    rng = np.random.default_rng(0)
    size = 64

    # Image constante
    constant = np.full((size, size), 3.0)
    assert f_statistic(constant) == 0.0

    # Regions homogenes avec valeurs differentes
    blocks = np.zeros((size, size))
    blocks[:32, :32] = 1.0
    blocks[:32, 32:] = 2.0
    blocks[32:, :32] = 3.0
    blocks[32:, 32:] = 4.0

    assert np.isinf(f_statistic(blocks))

    # Comparaison entre texture informative et bruit
    informative = rng.random((size, size)) * 0.1
    informative[:, 32:] += 5.0

    noise = rng.random((size, size))

    f_info = f_statistic(informative)
    f_noise = f_statistic(noise)

    assert f_info > f_noise

    print(f"F texture informative : {f_info:.2f}")
    print(f"F bruit : {f_noise:.2f}")

    # Selection parmi 12 caracteristiques
    stack = np.stack([
        informative if i % 3 == 0 else rng.random((size, size))
        for i in range(12)
    ])

    selected, indices, scores = select_features(stack)

    assert {0, 3, 6, 9}.issubset(indices)
    assert selected.shape[0] == len(indices)

    print(f"Scores F : {np.round(scores, 2)}")
    print(f"Caracteristiques conservees : {indices}")
    print(f"Dimensions finales : {selected.shape}")
    print("Tous les tests sont valides.")

    
