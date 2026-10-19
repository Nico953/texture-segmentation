
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def _validate_features(features, n_clusters):
    if features.ndim != 3 or features.size == 0:
        raise ValueError("features doit etre un tableau 3D non vide")

    if not np.isfinite(features).all():
        raise ValueError("features contient des valeurs invalides")

    n_pixels = features.shape[1] * features.shape[2]

    if not isinstance(n_clusters, (int, np.integer)):
        raise ValueError("n_clusters doit etre un entier")

    if not 2 <= n_clusters <= n_pixels:
        raise ValueError("Nombre de clusters invalide")


def smooth_features(features, window_size):
    """Lisse les cartes de caracteristiques."""

    return np.stack([
        uniform_filter(feature, size=window_size, mode="reflect")
        for feature in features
    ])


def standard_kmeans(
    features, n_clusters, smooth_window=None, random_state=0, n_init=10
):
    """Segmentation par K-means standard."""

    features = np.asarray(features, dtype=float)
    _validate_features(features, n_clusters)

    if smooth_window is not None:
        if (
            not isinstance(smooth_window, (int, np.integer))
            or smooth_window <= 0
            or smooth_window % 2 == 0
        ):
            raise ValueError("smooth_window doit etre un entier positif impair")

        features = smooth_features(features, smooth_window)

    c, h, w = features.shape
    vectors = features.reshape(c, -1).T

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )

    return model.fit_predict(vectors).reshape(h, w)


def _majority_vote_relabel(labels, window_size, n_clusters):
    """Attribue la classe majoritaire dans chaque voisinage."""

    counts = np.stack([
        uniform_filter(
            (labels == i).astype(float),
            size=window_size,
            mode="reflect"
        )
        for i in range(n_clusters)
    ])

    return np.argmax(counts, axis=0)


def _ambiguous_mask(labels, window_size):
    """Detecte les pixels proches de plusieurs classes."""

    local_min = minimum_filter(
        labels, size=window_size, mode="reflect"
    )
    local_max = maximum_filter(
        labels, size=window_size, mode="reflect"
    )

    return local_min != local_max


def _align_labels_to_reference(new_centers, new_labels, reference_centers):
    """Aligne les indices des clusters entre deux iterations."""

    distances = np.linalg.norm(
        new_centers[:, None, :] - reference_centers[None, :, :],
        axis=2
    )

    new_idx, ref_idx = linear_sum_assignment(distances)

    mapping = np.empty(len(reference_centers), dtype=int)
    mapping[new_idx] = ref_idx

    aligned_labels = mapping[new_labels]
    aligned_centers = new_centers[np.argsort(mapping)]

    return aligned_labels, aligned_centers


def iterative_kmeans(
    features,
    n_clusters,
    window_sizes=(33, 25, 17),
    random_state=0,
    n_init=10,
    recluster_ambiguous=True
):
    """Segmentation iterative avec reduction du voisinage."""

    features = np.asarray(features, dtype=float)
    _validate_features(features, n_clusters)

    if tuple(window_sizes) != (33, 25, 17):
        raise ValueError("window_sizes doit etre (33, 25, 17)")

    if not isinstance(recluster_ambiguous, bool):
        raise ValueError("recluster_ambiguous doit etre un booleen")

    c, h, w = features.shape

    labels = np.zeros((h, w), dtype=int)
    active = np.ones((h, w), dtype=bool)

    status_history = []
    reference_centers = None

    for window_size in window_sizes:

        smoothed = smooth_features(features, window_size)
        vectors = smoothed.reshape(c, -1).T
        active_flat = active.ravel()

        if active_flat.sum() >= n_clusters:

            if reference_centers is None or recluster_ambiguous:

                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=n_init
                )

                new_labels = model.fit_predict(vectors[active_flat])

                if reference_centers is None:
                    reference_centers = model.cluster_centers_
                else:
                    new_labels, reference_centers = (
                        _align_labels_to_reference(
                            model.cluster_centers_,
                            new_labels,
                            reference_centers
                        )
                    )

            else:
                distances = np.linalg.norm(
                    vectors[active_flat][:, None, :]
                    - reference_centers[None, :, :],
                    axis=2
                )
                new_labels = np.argmin(distances, axis=1)

            labels_flat = labels.ravel().copy()
            labels_flat[active_flat] = new_labels
            labels = labels_flat.reshape(h, w)

        # Fusion des petites regions
        labels = _majority_vote_relabel(
            labels, window_size, n_clusters
        )

        # Detection des pixels ambigus
        active = _ambiguous_mask(labels, window_size)
        status_history.append(active.copy())

    return labels, status_history


def compute_pcc(predicted_labels, ground_truth, n_clusters):
    """Calcule le pourcentage de pixels correctement classes."""

    predicted_labels = np.asarray(predicted_labels)
    ground_truth = np.asarray(ground_truth)

    if predicted_labels.ndim != 2 or predicted_labels.shape != ground_truth.shape:
        raise ValueError("Dimensions des segmentations invalides")

    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 2:
        raise ValueError("n_clusters invalide")

    for labels in (predicted_labels, ground_truth):
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Les etiquettes doivent etre des entiers")

        if labels.size == 0 or labels.min() < 0 or labels.max() >= n_clusters:
            raise ValueError("Etiquettes hors limites")

    # Matrice de correspondance entre clusters et classes
    contingency = np.zeros((n_clusters, n_clusters), dtype=int)

    np.add.at(
        contingency,
        (predicted_labels.ravel(), ground_truth.ravel()),
        1
    )

    pred_idx, true_idx = linear_sum_assignment(-contingency)

    mapping = dict(zip(pred_idx.tolist(), true_idx.tolist()))
    pcc = contingency[pred_idx, true_idx].sum() / predicted_labels.size

    return pcc, mapping


if __name__ == "__main__":

    rng = np.random.default_rng(0)

    # Deux textures synthetiques
    features = rng.normal(size=(3, 64, 64))
    features[:, :, 32:] += 4.0

    ground_truth = np.zeros((64, 64), dtype=int)
    ground_truth[:, 32:] = 1

    # K-means standard
    labels_standard = standard_kmeans(features, n_clusters=2)
    pcc_standard, _ = compute_pcc(
        labels_standard, ground_truth, 2
    )

    # K-means iteratif
    labels_iterative, history = iterative_kmeans(
        features, n_clusters=2
    )

    pcc_iterative, _ = compute_pcc(
        labels_iterative, ground_truth, 2
    )

    print(f"PCC standard : {pcc_standard:.2%}")
    print(f"PCC iteratif : {pcc_iterative:.2%}")

    for window, mask in zip((33, 25, 17), history):
        print(f"Fenetre {window} : {mask.sum()} pixels ambigus")

    assert labels_standard.shape == (64, 64)
    assert labels_iterative.shape == (64, 64)
    assert len(history) == 3
    assert 0 <= pcc_standard <= 1
    assert 0 <= pcc_iterative <= 1

    print("Tests termines.")
