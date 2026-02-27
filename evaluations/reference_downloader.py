"""
Reference File Download Utilities.

This module provides utilities for downloading and caching FID reference files
from URLs. It ensures files are validated and properly cached locally.
"""

import os
import urllib.request
import numpy as np
from typing import Optional


def download_reference_file(ref_url: str, cache_dir: str = None, verbose: bool = True) -> str:
    """
    Download FID reference statistics file from URL and cache locally.
    
    This function handles:
      - Automatic caching of downloaded files
      - Validation of downloaded NPZ files
      - Re-downloading if cached file is corrupted
      - Informative progress messages
    
    Args:
        ref_url: URL to download from
        cache_dir: Directory to cache the file (defaults to ./fid-refs/)
        verbose: Whether to print progress messages
    
    Returns:
        Local file path where the reference was saved
    
    Raises:
        RuntimeError: If download fails or file is corrupted
        ValueError: If ref_url is not a valid URL
    """
    if not ref_url.startswith('http://') and not ref_url.startswith('https://'):
        raise ValueError(f"Invalid URL: {ref_url}")
    
    if verbose:
        print(f"Downloading FID reference from {ref_url}...")
    
    # Extract filename from URL
    filename = os.path.basename(ref_url.split('?')[0])
    
    # Determine cache directory (fid-refs/ at project root)
    if cache_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "fid-refs")
    
    local_path = os.path.join(cache_dir, filename)
    
    # Check if file exists locally
    if os.path.exists(local_path):
        if verbose:
            print(f"Found cached reference at {local_path}")
        if _validate_npz_file(local_path, verbose):
            return local_path
        else:
            if verbose:
                print(f"Cached file is corrupted, re-downloading...")
            os.remove(local_path)
    
    # Download the file
    try:
        os.makedirs(cache_dir, exist_ok=True)
        
        if verbose:
            print(f"Downloading to {local_path}...")
        
        urllib.request.urlretrieve(ref_url, local_path)
        
        # Verify the downloaded file
        if not _validate_npz_file(local_path, verbose):
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError("Downloaded file is corrupted or invalid")
        
        if verbose:
            print(f"✓ Successfully downloaded and cached reference")
        
        return local_path
        
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise RuntimeError(f"Failed to download FID reference: {e}")


def _validate_npz_file(file_path: str, verbose: bool = False) -> bool:
    """
    Validate that a file is a valid NPZ file.
    
    Args:
        file_path: Path to the file to validate
        verbose: Whether to print error messages
    
    Returns:
        True if file is valid, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            np.load(f)
        return True
    except Exception as e:
        if verbose:
            print(f"File validation failed: {e}")
        return False


def is_url(path: str) -> bool:
    """
    Check if a path is a URL.
    
    Args:
        path: Path or URL string to check
    
    Returns:
        True if path is a URL, False otherwise
    """
    return path.startswith('http://') or path.startswith('https://')


def resolve_reference_path(ref_path: str, cache_dir: Optional[str] = None, 
                          verbose: bool = True) -> str:
    """
    Resolve a reference path, downloading if it's a URL.
    
    This is a convenience function that handles both URLs and local paths.
    
    Args:
        ref_path: Path or URL to the reference file
        cache_dir: Directory to cache downloaded files
        verbose: Whether to print progress messages
    
    Returns:
        Local file path to the reference
    
    Raises:
        FileNotFoundError: If local path doesn't exist
        RuntimeError: If URL download fails
    """
    if is_url(ref_path):
        return download_reference_file(ref_path, cache_dir, verbose)
    else:
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference file not found: {ref_path}")
        return ref_path

