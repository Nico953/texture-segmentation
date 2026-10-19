
import numpy as np
from scipy.signal import fftconvolve


def gabor_kernel(size, orientation_deg, freq, sigma_x=6.0, sigma_y=0.01):
    """Construit un filtre de Gabor reel non symetrique."""

    if not isinstance(size, (int, np.integer)) or size <= 0 or size % 2 == 0:
        raise ValueError("size doit etre un entier positif impair")
    if freq < 0 or sigma_x <= 0 or sigma_y <= 0:
        raise ValueError("Parametres du filtre invalides")

    half = size // 2
    y, x = np.meshgrid(
        np.arange(-half, half + 1),
        np.arange(-half, half + 1),
        indexing="ij"
    )

    phi = np.deg2rad(orientation_deg)

    q = np.cos(phi) * x + np.sin(phi) * y
    w = -np.sin(phi) * x + np.cos(phi) * y

    gaussian = np.exp(
        -(q ** 2) / (2 * sigma_x ** 2)
        - (w ** 2) / (2 * sigma_y ** 2)
    )

    return gaussian * np.cos(2 * np.pi * freq * q)


def build_filter_bank(
    size=31,
    orientations=(0, 45, 90, 135),
    freqs=(0.0, 0.05, 0.1),
    sigma_x=6.0,
    sigma_y=0.01
):
    """Construit les 12 filtres de Gabor."""

    kernels = []

    for orientation in orientations:
        for freq in freqs:
            kernel = gabor_kernel(
                size, orientation, freq, sigma_x, sigma_y
            )
            name = f"ori{orientation}_f{freq}"
            kernels.append((name, kernel))

    return kernels


def apply_filter_bank(image, kernels=None, **kernel_kwargs):
    """Applique les filtres de Gabor sur une image en niveaux de gris."""

    if kernels is None:
        kernels = build_filter_bank(**kernel_kwargs)

    image = np.asarray(image, dtype=float)
    filtered = np.empty((len(kernels), *image.shape), dtype=float)

    for i, (_, kernel) in enumerate(kernels):
        pad = kernel.shape[0] // 2
        padded = np.pad(image, pad, mode="reflect")
        filtered[i] = fftconvolve(padded, kernel, mode="valid")

    names = [name for name, _ in kernels]
    return filtered, names


def _make_oriented_stripes(size, angle_deg, period=8):
    """Genere des rayures sinusoidales pour tester les filtres."""

    y, x = np.meshgrid(
        np.arange(size),
        np.arange(size),
        indexing="ij"
    )

    phi = np.deg2rad(angle_deg)
    projection = -np.sin(phi) * x + np.cos(phi) * y

    return 0.5 + 0.5 * np.sin(2 * np.pi * projection / period)


if __name__ == "__main__":

    kernels = build_filter_bank()
    print(f"Nombre de filtres : {len(kernels)}")

    image = np.random.default_rng(0).random((64, 64))
    filtered, names = apply_filter_bank(image, kernels)

    assert len(kernels) == 12
    assert filtered.shape == (12, 64, 64)
    assert np.isfinite(filtered).all()

    print(f"Dimensions des images filtrees : {filtered.shape}")

    # Test des parametres
    try:
        gabor_kernel(10, 0, 0.05)
    except ValueError:
        print("Test taille paire : OK")
    else:
        raise AssertionError("Taille paire non rejetee")

    # Test de reponse aux orientations
    print("\nReponse des filtres selon les orientations :")

    for angle in (0, 45, 90, 135):
        texture = _make_oriented_stripes(96, angle)
        responses, names = apply_filter_bank(texture, kernels)

        energies = {
            name: float(np.sum(response ** 2))
            for name, response in zip(names, responses)
            if "f0.0" not in name
        }

        best_filter = max(energies, key=energies.get)
        print(f"Texture {angle} degres : {best_filter}")
