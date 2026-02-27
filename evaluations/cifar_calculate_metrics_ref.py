#!/usr/bin/env python
"""
Generate CIFAR-10 reference file for calculate_metrics (FID + FD_DINOv2).

Creates a .pkl file with both fid and fd_dinov2 statistics by running
calculate_metrics ref on CIFAR-10 training images. Used when the default
npz reference (FID only) is insufficient.
"""

import os
import sys
import tempfile
import zipfile
import numpy as np
from PIL import Image

from .cifar_reference import load_cifar10_train_raw


def _create_cifar10_zip(images: np.ndarray, zip_path: str) -> None:
    """Save CIFAR-10 images to a zip file for ImageFolderDataset."""
    # images: (N, H, W, C) uint8
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for i in range(len(images)):
            img = Image.fromarray(images[i])
            buf = _pil_to_png_bytes(img)
            zf.writestr(f"{i:05d}.png", buf)


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to PNG bytes."""
    import io
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def generate_cifar10_calculate_metrics_ref(
    output_path: str,
    data_root: str = "./data",
    force_regenerate: bool = False,
) -> str:
    """
    Generate CIFAR-10 reference file with both FID and FD_DINOv2 statistics.

    Creates a .pkl file by running calculate_metrics ref on CIFAR-10 training
    images. Use this reference to compute both metrics (npz only has FID).

    Args:
        output_path: Path where the .pkl file will be saved
        data_root: Root directory for CIFAR-10 data
        force_regenerate: If True, regenerate even if file exists

    Returns:
        Path to the reference file
    """
    if os.path.exists(output_path) and not force_regenerate:
        print(f"Reference file already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print("Loading CIFAR-10 training set...")
    images = load_cifar10_train_raw(data_root)

    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        zip_path = tmp.name

    try:
        print("Creating temporary zip with CIFAR-10 images...")
        _create_cifar10_zip(images, zip_path)

        print("Computing FID and FD_DINOv2 reference statistics...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + (os.pathsep + env.get("PYTHONPATH", ""))
        cmd = [
            sys.executable,
            "-m", "evaluations.calculate_metrics",
            "ref",
            "--data", zip_path,
            "--dest", output_path,
            "--metrics", "fid,fd_dinov2",
        ]
        import subprocess
        subprocess.run(cmd, check=True, cwd=project_root, env=env)

        print(f"Reference saved to {output_path}")
        return output_path
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
