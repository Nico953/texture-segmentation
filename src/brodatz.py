from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_texture(path, size=128):
    """
    Load a grayscale texture and extract a centered square.
    """
    image = Image.open(path).convert("L")
    image = np.asarray(image, dtype=np.float64)

    height, width = image.shape

    if height < size or width < size:
        raise ValueError(f"Texture too small: {path}")

    y = (height - size) // 2
    x = (width - size) // 2

    return image[y:y + size, x:x + size]


def create_mosaic(texture_paths, size=128):
    """
    Create a 2x2 texture mosaic and its ground truth.
    """
    if len(texture_paths) != 4:
        raise ValueError("Exactly 4 textures are required")

    textures = [
        load_texture(path, size)
        for path in texture_paths
    ]

    mosaic = np.block([
        [textures[0], textures[1]],
        [textures[2], textures[3]]
    ])

    ground_truth = np.block([
        [
            np.zeros((size, size), dtype=np.uint8),
            np.ones((size, size), dtype=np.uint8)
        ],
        [
            np.full((size, size), 2, dtype=np.uint8),
            np.full((size, size), 3, dtype=np.uint8)
        ]
    ])

    return mosaic, ground_truth


def display_mosaic(mosaic, ground_truth):
    """
    Display the mosaic and its ground truth.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(mosaic, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Brodatz mosaic")

    axes[1].imshow(
        ground_truth,
        cmap="tab10",
        vmin=0,
        vmax=3,
        interpolation="nearest"
    )
    axes[1].set_title("Ground truth")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "brodatz"

    texture_paths = [
        data_dir / "D9.png",
        data_dir / "D15.png",
        data_dir / "D29.png",
        data_dir / "D68.png"
    ]

    mosaic, ground_truth = create_mosaic(texture_paths)

    print("Mosaic shape:", mosaic.shape)
    print("Ground truth shape:", ground_truth.shape)
    print("Classes:", np.unique(ground_truth))

    display_mosaic(mosaic, ground_truth)