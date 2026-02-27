"""
PyTorch-based evaluator for generative model metrics.

This module provides evaluation metrics (IS, FID, sFID, Precision, Recall) without
TensorFlow or protobuf dependencies. It uses the same Inception model as fid.py
(inception-2015-12-05.pkl) for feature extraction to ensure consistency with
the project's FID implementation.

Compatible with the same NPZ input format as evaluator.py (ref_batch, sample_batch).
"""

import argparse
import io
import os
import pickle
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from tqdm.auto import tqdm

# Constants
POOL_FEATURE_DIM = 2048
SPATIAL_CHANNELS = 7  # First N channels from mixed_6 for sFID (matches TF evaluator)
INCEPTION_SCORE_SPLIT_SIZE = 5000


def main() -> None:
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate generative model metrics (IS, FID, sFID, Precision, Recall)"
    )
    parser.add_argument("ref_batch", help="Path to reference batch npz file")
    parser.add_argument("sample_batch", help="Path to sample batch npz file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Default: cuda if available",
    )
    args = parser.parse_args()

    use_gpu = os.environ.get("USE_GPU_EVALUATOR", "1") == "1"
    device = _get_device(args.device, use_gpu)

    evaluator = PyTorchEvaluator(device=device, batch_size=args.batch_size)

    print("Warming up PyTorch evaluator...")
    evaluator.warmup()

    print("Computing reference batch activations...")
    ref_acts = evaluator.read_activations(args.ref_batch)

    print("Computing/reading reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

    print("Computing sample batch activations...")
    sample_acts = evaluator.read_activations(args.sample_batch)
    print("Computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(
        args.sample_batch, sample_acts
    )

    print("Computing evaluations...")
    inception_score = evaluator.compute_inception_score(sample_acts[0])
    fid = sample_stats.frechet_distance(ref_stats)
    sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    precision, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

    print("Inception Score:", inception_score)
    print("FID:", fid)
    print("sFID:", sfid)
    print("Precision:", precision)
    print("Recall:", recall)


def _get_device(device_str: Optional[str], use_gpu: bool) -> torch.device:
    """Resolve PyTorch device from string and environment."""
    if device_str:
        return torch.device(device_str)
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# FID Statistics
# -----------------------------------------------------------------------------


class FIDStatistics:
    """Holds mean and covariance for Frechet distance computation."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other: "FIDStatistics", eps: float = 1e-6) -> float:
        """Compute Frechet distance between two Gaussian distributions."""
        mu1, sigma1 = np.atleast_1d(self.mu), np.atleast_2d(self.sigma)
        mu2, sigma2 = np.atleast_1d(other.mu), np.atleast_2d(other.sigma)

        if mu1.shape != mu2.shape:
            raise ValueError(
                f"Mean vectors have different shapes: {mu1.shape} vs {mu2.shape}"
            )
        if sigma1.shape != sigma2.shape:
            raise ValueError(
                f"Covariances have different shapes: {sigma1.shape} vs {sigma2.shape}"
            )

        diff = mu1 - mu2
        result = linalg.sqrtm(sigma1.dot(sigma2))
        covmean = result[0] if isinstance(result, tuple) else result

        if not np.isfinite(covmean).all():
            warnings.warn(
                f"FID calculation produces singular product; adding {eps} to diagonal"
            )
            offset = np.eye(sigma1.shape[0]) * eps
            result = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            covmean = result[0] if isinstance(result, tuple) else result

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(
                    f"Imaginary component {np.max(np.abs(covmean.imag))}"
                )
            covmean = covmean.real

        return float(
            diff.dot(diff)
            + np.trace(sigma1)
            + np.trace(sigma2)
            - 2 * np.trace(covmean)
        )


# -----------------------------------------------------------------------------
# Inception Feature Extractor (Pool + Spatial + Logits for IS)
# Uses inception-2015-12-05.pkl - same model as fid.py for consistency
# -----------------------------------------------------------------------------


def _get_local_inception_pkl_path() -> str:
    """Return path to pretrain_models/inception-2015-12-05.pkl (download if missing)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkl_path = os.path.join(project_root, "pretrain_models", "inception-2015-12-05.pkl")
    if not os.path.exists(pkl_path):
        _ensure_inception_model(project_root)
    return pkl_path


def _ensure_inception_model(project_root: str) -> None:
    """Download inception-2015-12-05.pkl to pretrain_models/ if missing."""
    import urllib.request
    pkl_path = os.path.join(project_root, "pretrain_models", "inception-2015-12-05.pkl")
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    print(f"Downloading Inception model to {pkl_path}...")
    try:
        import dnnlib
        with dnnlib.util.open_url(url, verbose=True) as f:
            with open(pkl_path, "wb") as out:
                out.write(f.read())
        print("Download complete.")
    except Exception:
        urllib.request.urlretrieve(url, pkl_path)


def _load_inception_pkl_model(device: torch.device):
    """Load inception-2015-12-05.pkl model (same as fid.py)."""
    pkl_path = _get_local_inception_pkl_path()
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    model = model.to(device)
    model.eval()
    return model


class InceptionFeatureExtractor(nn.Module):
    """
    Extracts pool, spatial (mixed_6), and logits using inception-2015-12-05.pkl.
    Same model as fid.py for FID/IS consistency.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self._model = _load_inception_pkl_model(device)
        # Logits layer: model has 'output' (Linear 2048->1008) for Inception Score
        self._logits_layer = self._model.output if hasattr(
            self._model, "output"
        ) else getattr(self._model, "output", None)
        if self._logits_layer is None:
            raise RuntimeError("Inception model has no output layer for logits")

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pool features, spatial features, and softmax logits.

        Uses same input format as fid.py and calculate_metrics.py: NCHW, [0, 255].
        Do NOT normalize to [0,1] or [-1,1] - the pkl model expects [0,255].

        Args:
            images: NHWC float32 in [0, 255], or NCHW in [0, 255]

        Returns:
            pool_features: (N, 2048)
            spatial_features: (N, 7*H*W) for sFID
            softmax_probs: (N, 1008) for Inception Score
        """
        # Ensure NCHW and [0, 255] range (matches fid.py, calculate_metrics.py)
        if images.dim() == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.max() <= 1.0:
            images = images.float() * 255.0
        images = images.to(self.device).float()

        # Capture mixed_6 output for spatial features (sFID)
        spatial_out = [None]

        def hook_fn(_module, _input, output):
            spatial_out[0] = output

        handle = self._model.layers.mixed_6.register_forward_hook(hook_fn)
        try:
            pool = self._model(images, return_features=True)
        finally:
            handle.remove()

        spatial = spatial_out[0]  # [N, 768, 17, 17]
        # Spatial: take first 7 channels, flatten (matches TF mixed_6/conv:0[..., :7])
        spatial_flat = spatial[:, :SPATIAL_CHANNELS].reshape(
            spatial.shape[0], -1
        ).cpu().numpy()

        pool_flat = pool.cpu().numpy()

        # Logits for Inception Score
        pool_tensor = pool
        logits = self._logits_layer(pool_tensor)
        softmax_probs = F.softmax(logits, dim=1).cpu().numpy()

        return pool_flat, spatial_flat, softmax_probs


# -----------------------------------------------------------------------------
# NumPy-based Distance Block (replaces TensorFlow DistanceBlock)
# -----------------------------------------------------------------------------


def _pairwise_squared_distances(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances: ||u - v||^2."""
    norm_u = np.sum(u**2, axis=1, keepdims=True)
    norm_v = np.sum(v**2, axis=1, keepdims=True)
    dist = np.maximum(
        norm_u - 2 * (u @ v.T) + norm_v.T, 0.0
    ).astype(np.float32)
    return dist


def _less_thans(
    batch_1: np.ndarray,
    radii_1: np.ndarray,
    batch_2: np.ndarray,
    radii_2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Check which samples fall within manifold radii (for Precision/Recall)."""
    dist = _pairwise_squared_distances(batch_1, batch_2)  # [B1, B2]
    # batch_1_in: for each sample in batch_1, is it within any batch_2 radius?
    # dist[i,j] <= radii_2[j,k] for any j -> [B1, K]
    batch_1_in = np.any(
        dist[:, :, np.newaxis] <= radii_2[np.newaxis, :, :], axis=1
    )
    # batch_2_in: for each sample in batch_2, is it within any batch_1 radius?
    # dist[i,j] <= radii_1[i,k] for any i -> [B2, K]
    batch_2_in = np.any(
        dist[:, :, np.newaxis] <= radii_1[:, np.newaxis, :], axis=0
    )
    return batch_1_in, batch_2_in


# -----------------------------------------------------------------------------
# Manifold Estimator (NumPy implementation)
# -----------------------------------------------------------------------------


class ManifoldEstimator:
    """
    Estimates manifold of feature vectors for Precision/Recall computation.
    Uses NumPy instead of TensorFlow.
    """

    def __init__(
        self,
        row_batch_size: int = 10000,
        col_batch_size: int = 10000,
        nhood_sizes: Tuple[int, ...] = (3,),
        clamp_to_percentile: Optional[float] = None,
        eps: float = 1e-5,
    ) -> None:
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        """Compute k-NN radii for each sample."""
        num_images = len(features)
        radii = np.zeros((num_images, self.num_nhoods), dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            distance_batch = np.zeros(
                (end1 - begin1, num_images), dtype=np.float32
            )

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]
                distance_batch[:, begin2:end2] = _pairwise_squared_distances(
                    row_batch, col_batch
                )

            partitioned = _numpy_partition(
                distance_batch, kth=seq, axis=1
            )
            radii[begin1:end1, :] = np.concatenate(
                [x[:, self.nhood_sizes] for x in partitioned],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(
                radii, self.clamp_to_percentile, axis=0
            )
            radii = np.minimum(radii, max_distances)
        return radii

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Precision and Recall."""
        features_1_status = np.zeros(
            (len(features_1), radii_2.shape[1]), dtype=bool
        )
        features_2_status = np.zeros(
            (len(features_2), radii_1.shape[1]), dtype=bool
        )

        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = min(begin_1 + self.row_batch_size, len(features_1))
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = min(begin_2 + self.col_batch_size, len(features_2))
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = _less_thans(
                    batch_1,
                    radii_1[begin_1:end_1],
                    batch_2,
                    radii_2[begin_2:end_2],
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in

        precision = np.mean(features_2_status.astype(np.float64), axis=0)
        recall = np.mean(features_1_status.astype(np.float64), axis=0)
        return precision, recall


def _numpy_partition(
    arr: np.ndarray, kth: np.ndarray, **kwargs
) -> list:
    """Parallel partition for k-NN (matches evaluator.py)."""
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers
    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size
    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))


# -----------------------------------------------------------------------------
# NPZ Array Readers (from evaluator.py, no TF)
# -----------------------------------------------------------------------------


class NpzArrayReader(ABC):
    """Abstract reader for NPZ array data."""

    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def remaining(self) -> int:
        pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch

        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return _BatchIterator(gen_fn, num_batches)


class _BatchIterator:
    def __init__(self, gen_fn, length: int) -> None:
        self._gen_fn = gen_fn
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterable:
        return self._gen_fn()


class StreamingNpzArrayReader(NpzArrayReader):
    """Streaming reader for large NPZ arrays."""

    def __init__(self, arr_f, shape, dtype) -> None:
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None
        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs
        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)
        read_count = bs * int(np.prod(self.shape[1:]))
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self) -> int:
        return max(0, self.shape[0] - self.idx)


class MemoryNpzArrayReader(NpzArrayReader):
    """In-memory reader for NPZ arrays."""

    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str) -> "MemoryNpzArrayReader":
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None
        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int:
        return max(0, self.arr.shape[0] - self.idx)


def _read_bytes(
    fp, size: int, error_template: str = "ran out of data"
) -> bytes:
    """Read exactly size bytes from file-like object."""
    data = bytes()
    while True:
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        raise ValueError(
            f"EOF: {error_template}, expected {size} got {len(data)}"
        )
    return data


@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f


@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    """Open NPZ array for streaming or in-memory read."""
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


# -----------------------------------------------------------------------------
# PyTorch Evaluator
# -----------------------------------------------------------------------------


class PyTorchEvaluator:
    """
    Evaluator for IS, FID, sFID, Precision, Recall using PyTorch only.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        softmax_batch_size: int = 512,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self._feature_extractor = InceptionFeatureExtractor(self.device)
        self._manifold_estimator = ManifoldEstimator()

    def warmup(self) -> None:
        """Run a dummy batch to initialize CUDA/cache."""
        dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
        self.compute_activations(iter([dummy]))

    def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load activations from NPZ file."""
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(
        self, batches: Iterable[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pool and spatial features from image batches.

        Args:
            batches: Iterator of NHWC numpy arrays in [0, 255]

        Returns:
            (pool_features, spatial_features) as numpy arrays
        """
        pool_list = []
        spatial_list = []

        for batch in tqdm(batches, desc="Computing activations"):
            batch = batch.astype(np.float32)
            # NHWC [0,255] -> NCHW [0,255] (matches fid.py, calculate_metrics)
            batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2)

            pool, spatial, _ = self._feature_extractor(batch_t)
            pool_list.append(pool)
            spatial_list.append(spatial)

        return (
            np.concatenate(pool_list, axis=0),
            np.concatenate(spatial_list, axis=0),
        )

    def read_statistics(
        self, npz_path: str, activations: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[FIDStatistics, FIDStatistics]:
        """Compute (mu, sigma) from activations. Never use precomputed stats from
        NPZ (e.g. from TF evaluator) to avoid mixing different Inception models."""
        return tuple(
            self._compute_statistics(x) for x in activations
        )

    def _compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(
        self, activations: np.ndarray, split_size: int = INCEPTION_SCORE_SPLIT_SIZE
    ) -> float:
        """Compute Inception Score from pool activations via softmax."""
        softmax_list = []
        for i in range(0, len(activations), self.softmax_batch_size):
            batch = activations[i : i + self.softmax_batch_size]
            batch_t = torch.from_numpy(batch.astype(np.float32)).to(self.device)
            with torch.no_grad():
                logits = self._feature_extractor._logits_layer(batch_t)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            softmax_list.append(probs)
        preds = np.concatenate(softmax_list, axis=0)

        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            p_y = np.mean(part, axis=0, keepdims=True)
            kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))

    def compute_prec_recall(
        self,
        activations_ref: np.ndarray,
        activations_sample: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute Precision and Recall between reference and sample manifolds."""
        radii_ref = self._manifold_estimator.manifold_radii(activations_ref)
        radii_sample = self._manifold_estimator.manifold_radii(activations_sample)
        precision, recall = self._manifold_estimator.evaluate_pr(
            activations_ref, radii_ref, activations_sample, radii_sample
        )
        return float(precision[0]), float(recall[0])


if __name__ == "__main__":
    main()
