
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter


def _variation(image, eps):
    """Calcule la variation locale max - min."""

    size = 2 * eps + 1
    local_max = maximum_filter(image, size=size, mode="reflect")
    local_min = minimum_filter(image, size=size, mode="reflect")

    return local_max - local_min


def local_fractal_dimension(
    image, window_size=16, eps_values=(1, 2), flat_threshold=1e-6
):
    """Calcule la dimension fractale locale avec la methode de variation."""

    image = np.asarray(image, dtype=float)

    if image.ndim != 2 or image.size == 0:
        raise ValueError("image doit etre une image 2D non vide")
    if not np.isfinite(image).all():
        raise ValueError("image contient des valeurs invalides")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size doit etre un entier positif")
    if len(eps_values) != 2:
        raise ValueError("Deux valeurs de eps sont necessaires")

    eps1, eps2 = eps_values

    if not all(isinstance(e, int) and e > 0 for e in eps_values):
        raise ValueError("eps_values doit contenir deux entiers positifs")
    if eps1 == eps2:
        raise ValueError("Les deux echelles doivent etre distinctes")
    if window_size < 2 * max(eps_values) + 1:
        raise ValueError("window_size est trop petit")
    if not 0 < flat_threshold < 1:
        raise ValueError("flat_threshold doit etre entre 0 et 1")

    image_range = np.ptp(image)

    if image_range == 0:
        return np.full(image.shape, 2.0)

    # Variation locale aux deux echelles
    e1 = uniform_filter(
        _variation(image, eps1), size=window_size, mode="reflect"
    )
    e2 = uniform_filter(
        _variation(image, eps2), size=window_size, mode="reflect"
    )

    # Gestion des zones plates et des valeurs proches de zero
    threshold = flat_threshold * image_range
    flat_mask = (e1 < threshold) & (e2 < threshold)

    floor = threshold * 1e-6
    e1 = np.maximum(e1, floor)
    e2 = np.maximum(e2, floor)

    # Pente log-log de la methode de variation
    fd = 3 - np.log(e2 / e1) / np.log(eps2 / eps1)

    return np.where(flat_mask, 2.0, fd)


def fractal_feature_stack(filtered_images, window_size=16, eps_values=(1, 2)):
    """Calcule la FD pour chaque image filtree par Gabor."""

    filtered_images = np.asarray(filtered_images, dtype=float)

    if filtered_images.ndim != 3 or filtered_images.size == 0:
        raise ValueError("filtered_images doit etre un tableau 3D non vide")

    features = np.empty_like(filtered_images, dtype=float)

    for i, image in enumerate(filtered_images):
        features[i] = local_fractal_dimension(
            image,
            window_size=window_size,
            eps_values=eps_values
        )

    return features


if __name__ == "__main__":

    rng = np.random.default_rng(0)
    size = 64

    x = np.arange(size)

    textures = {
        "Image constante": np.full((size, size), 0.5),
        "Bruit uniforme": rng.random((size, size)),
        "Rayures fines": np.tile(
            0.5 + 0.5 * np.sin(2 * np.pi * x / 4), (size, 1)
        ),
        "Rayures larges": np.tile(
            0.5 + 0.5 * np.sin(2 * np.pi * x / 32), (size, 1)
        )
    }

    print("Resultats de dimension fractale :")

    for name, image in textures.items():
        fd = local_fractal_dimension(image)
        print(
            f"{name:20s} "
            f"min={fd.min():.3f} "
            f"max={fd.max():.3f} "
            f"moyenne={fd.mean():.3f}"
        )

    # Verification des proprietes principales
    fd_flat = local_fractal_dimension(textures["Image constante"])
    assert np.allclose(fd_flat, 2.0)

    fd_noise = local_fractal_dimension(textures["Bruit uniforme"])
    fd_stripes = local_fractal_dimension(textures["Rayures larges"])

    assert fd_noise.mean() > fd_stripes.mean()

    for scale in (1e-3, 1e-5, 1e-8):
        fd_scaled = local_fractal_dimension(
            textures["Bruit uniforme"] * scale
        )
        assert np.allclose(fd_noise, fd_scaled, atol=1e-6)

    print("\nTous les tests sont valides.")
