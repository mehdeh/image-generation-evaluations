# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
import subprocess
import urllib.request
from torch_utils import distributed as dist
from src.datasets import dataset

#----------------------------------------------------------------------------

def download_inception_model(local_path, verbose=True):
    """
    Download the Inception-v3 model with multiple fallback options.
    
    Args:
        local_path: Local file path where the model should be saved
        verbose: Whether to print progress messages
    
    Returns:
        True if download was successful, False otherwise
    """
    # Primary URL
    primary_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    
    # Alternative URL for wget
    alternative_url = 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=metrics/inception-2015-12-05.pkl'
    
    # Manual download instructions
    manual_instructions = """
    
    ================================================================================
    AUTOMATIC DOWNLOAD FAILED
    ================================================================================
    Please manually download the Inception model:
    
    1. Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files?version=1
    2. Navigate to: File Browser > metrics
    3. Download: inception-2015-12-05.pkl
    4. Place the file in: {local_dir}
    
    After downloading, please run the command again.
    ================================================================================
    """.format(local_dir=os.path.dirname(local_path))
    
    # Try primary URL with dnnlib
    if verbose:
        print(f"Attempting to download Inception model from primary source...")
    try:
        with dnnlib.util.open_url(primary_url, verbose=verbose) as f:
            with open(local_path, 'wb') as out_file:
                out_file.write(f.read())
        if verbose:
            print(f"Successfully downloaded Inception model to {local_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"Primary download failed: {e}")
    
    # Try alternative URL with wget
    if verbose:
        print(f"Attempting to download from alternative source using wget...")
    try:
        result = subprocess.run(
            ['wget', '--content-disposition', alternative_url, '-O', local_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0 and os.path.exists(local_path):
            if verbose:
                print(f"Successfully downloaded Inception model to {local_path}")
            return True
        else:
            if verbose:
                print(f"Wget download failed: {result.stderr}")
    except Exception as e:
        if verbose:
            print(f"Wget download failed: {e}")
    
    # Try direct urllib download
    if verbose:
        print(f"Attempting direct download...")
    try:
        urllib.request.urlretrieve(primary_url, local_path)
        if os.path.exists(local_path):
            if verbose:
                print(f"Successfully downloaded Inception model to {local_path}")
            return True
    except Exception as e:
        if verbose:
            print(f"Direct download failed: {e}")
    
    # All methods failed
    if verbose:
        print(manual_instructions)
    return False


def load_inception_model(device, verbose=True):
    """
    Load the Inception-v3 model, downloading if necessary.
    
    This function checks for the model locally first, then attempts
    to download it using multiple fallback methods if not found.
    
    Args:
        device: PyTorch device to load the model on
        verbose: Whether to print progress messages
    
    Returns:
        The loaded Inception-v3 model
    
    Raises:
        FileNotFoundError: If model cannot be loaded or downloaded
    """
    # Project root: directory containing evaluations/ and pretrain_models/
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    # Local path for the Inception model
    local_dir = os.path.join(project_root, 'pretrain_models')
    local_path = os.path.join(local_dir, 'inception-2015-12-05.pkl')
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Check if model exists locally
    if os.path.exists(local_path):
        if verbose:
            print(f"Found local Inception model at {local_path}")
        try:
            with open(local_path, 'rb') as f:
                return pickle.load(f).to(device)
        except Exception as e:
            if verbose:
                print(f"Failed to load local model: {e}")
                print(f"Attempting to re-download...")
            # Remove corrupted file
            os.remove(local_path)
    
    # Model not found locally, attempt download
    if verbose:
        print(f"Inception model not found locally. Attempting download...")
    
    if download_inception_model(local_path, verbose):
        try:
            with open(local_path, 'rb') as f:
                return pickle.load(f).to(device)
        except Exception as e:
            raise RuntimeError(f"Downloaded model is corrupted: {e}")
    else:
        raise FileNotFoundError(
            f"Failed to download Inception model. Please download manually and place in {local_dir}"
        )


def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    """
    Calculate Inception statistics for a set of images.
    
    Args:
        image_path: Path to the directory containing images
        num_expected: Expected number of images (optional)
        seed: Random seed for image selection
        max_batch_size: Maximum batch size for processing
        num_workers: Number of worker threads for data loading
        prefetch_factor: Prefetch factor for data loading
        device: PyTorch device to use
    
    Returns:
        Tuple of (mu, sigma) statistics
    """
    # Rank 0 goes first (load model so other ranks can proceed after barrier).
    if dist.get_rank() != 0:
        dist.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    detector_net = load_inception_model(device, verbose=(dist.get_rank() == 0))

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        dist.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        dist.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals (aggregate across ranks when running distributed).
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(mu)
        torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate Frechet Inception Distance (FID).

    Examples:

    \b
    # Generate 50000 images and save them as fid-tmp/*/*.png
    torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Calculate FID
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

    \b
    # Compute dataset reference statistics
    python fid.py ref --data=datasets/my-dataset.zip --dest=fid-refs/my-dataset.npz
    """

#----------------------------------------------------------------------------

def download_fid_reference(ref_url, local_path, verbose=True):
    """
    Download FID reference statistics file.
    
    Args:
        ref_url: URL to download from
        local_path: Local file path where the reference should be saved
        verbose: Whether to print progress messages
    
    Returns:
        True if download was successful, False otherwise
    """
    if verbose:
        print(f"Downloading FID reference from {ref_url}...")
    
    try:
        with dnnlib.util.open_url(ref_url, verbose=verbose) as f:
            data = dict(np.load(f))
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            # Save to local file
            np.savez(local_path, **data)
        if verbose:
            print(f"Successfully downloaded FID reference to {local_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"Failed to download FID reference: {e}")
        return False


def load_fid_reference(ref_path, verbose=True):
    """
    Load FID reference statistics, downloading if necessary.
    
    Args:
        ref_path: Path or URL to the FID reference file
        verbose: Whether to print progress messages
    
    Returns:
        Dictionary containing FID reference statistics
    """
    # If it's a URL, try to download it
    if ref_path.startswith('http://') or ref_path.startswith('https://'):
        # Extract filename from URL
        filename = os.path.basename(ref_path.split('?')[0])
        
        # Project root: directory containing evaluations/ and fid-refs/
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        
        # Local path for the FID reference
        local_dir = os.path.join(project_root, 'fid-refs')
        local_path = os.path.join(local_dir, filename)
        
        # Check if file exists locally
        if os.path.exists(local_path):
            if verbose:
                print(f"Found local FID reference at {local_path}")
            try:
                with open(local_path, 'rb') as f:
                    return dict(np.load(f))
            except Exception as e:
                if verbose:
                    print(f"Failed to load local FID reference: {e}")
                    print(f"Attempting to re-download...")
                os.remove(local_path)
        
        # Download the file
        if download_fid_reference(ref_path, local_path, verbose):
            try:
                with open(local_path, 'rb') as f:
                    return dict(np.load(f))
            except Exception as e:
                raise RuntimeError(f"Downloaded FID reference is corrupted: {e}")
        else:
            # Fall back to direct loading
            if verbose:
                print(f"Falling back to direct loading from URL...")
            with dnnlib.util.open_url(ref_path) as f:
                return dict(np.load(f))
    else:
        # Local file
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"FID reference file not found: {ref_path}")
        
        with open(ref_path, 'rb') as f:
            return dict(np.load(f))


@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)

def calc(image_path, ref_path, num_expected, seed, batch):
    """Calculate FID for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        ref = load_fid_reference(ref_path, verbose=True)

    mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
    dist.print0('Calculating FID...')
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        print(f'{fid:g}')
    dist.barrier()

#----------------------------------------------------------------------------

@main.command()
@click.option('--data', 'dataset_path', help='Path to the dataset', metavar='PATH|ZIP', type=str, required=True)
@click.option('--dest', 'dest_path',    help='Destination .npz file', metavar='NPZ',    type=str, required=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',       type=click.IntRange(min=1), default=64, show_default=True)

def ref(dataset_path, dest_path, batch):
    """Calculate dataset reference statistics needed by 'calc'."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch)
    dist.print0(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.get_rank() == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    dist.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
