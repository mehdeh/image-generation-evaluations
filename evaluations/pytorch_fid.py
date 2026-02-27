"""
PyTorch-FID Evaluation Method.

This module provides FID (Fréchet Inception Distance) evaluation using the
official pytorch-fid library, which is widely used in the research community.

The reference can be specified as:
  - A local file path (e.g., ./fid-refs/cifar10-32x32.npz)
  - A URL (e.g., https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz)

When a URL is provided, it will be downloaded and cached automatically by
the evaluate.py wrapper before being passed to pytorch-fid.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any


def run_pytorch_fid_evaluation(exp_dir: str, config: Dict[str, Any], logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Run FID evaluation using the official pytorch-fid library.
    
    This method uses the popular pytorch-fid package which provides
    a standard FID implementation used widely in the research community.
    It requires:
      - Generated images in: exp_dir/samples/
      - Reference path specified in config['evaluation']['pytorch_fid_ref']
        Can be either:
        1. A directory containing reference images (e.g., CIFAR-10 training set)
        2. An NPZ file with precomputed statistics (mu, sigma)
    
    Args:
        exp_dir: Experiment directory path
        config: Configuration dictionary with evaluation settings
        logger: Optional logger instance (creates new one if None)
    
    Returns:
        Dictionary containing FID score and metadata
    """
    if logger is None:
        logger = logging.getLogger('pytorch_fid')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
    
    logger.info("="*70)
    logger.info("PyTorch-FID Evaluation (Official Implementation)")
    logger.info("="*70)
    
    eval_config = config.get('evaluation', {})
    images_dir = os.path.join(exp_dir, 'samples')
    pytorch_fid_ref = eval_config.get('pytorch_fid_ref')
    
    # Validate generated images directory
    if not os.path.exists(images_dir):
        error_msg = f"Images directory not found: {images_dir}"
        logger.error(error_msg)
        return {'error': error_msg, 'method': 'pytorch_fid'}
    
    # Check and validate reference directory
    if not pytorch_fid_ref:
        error_msg = "PyTorch-FID reference directory not configured"
        logger.error(error_msg)
        logger.error("")
        logger.error("To use PyTorch-FID evaluation, you need to configure the reference directory:")
        logger.error("1. Add 'pytorch_fid_ref' to your config file under 'evaluation' section")
        logger.error("2. Point it to a directory containing reference images (e.g., CIFAR-10 training set)")
        logger.error("")
        logger.error("Example configuration in config.yaml:")
        logger.error("  evaluation:")
        logger.error("    pytorch_fid_ref: './datasets/cifar10_train/'")
        logger.error("")
        logger.error("Alternatively, you can use the custom FID method (--fid) which uses NPZ statistics files.")
        return {'error': error_msg, 'method': 'pytorch_fid', 'config_missing': True}
    
    # Validate reference path exists
    if not os.path.exists(pytorch_fid_ref):
        error_msg = f"PyTorch-FID reference path not found: {pytorch_fid_ref}"
        logger.error(error_msg)
        logger.error("")
        logger.error(f"The configured reference path does not exist: {pytorch_fid_ref}")
        logger.error("Please check your configuration and ensure the path is correct.")
        logger.error("")
        logger.error("For CIFAR-10 evaluation, you can use:")
        logger.error("  1. A directory containing the training images")
        logger.error("  2. An NPZ file with precomputed statistics (mu, sigma)")
        return {'error': error_msg, 'method': 'pytorch_fid', 'path_invalid': True}
    
    # Check if it's an NPZ file or directory
    is_npz_file = pytorch_fid_ref.endswith('.npz') and os.path.isfile(pytorch_fid_ref)
    is_directory = os.path.isdir(pytorch_fid_ref)
    
    if not is_npz_file and not is_directory:
        error_msg = f"PyTorch-FID reference path must be a directory or NPZ file: {pytorch_fid_ref}"
        logger.error(error_msg)
        logger.error("")
        logger.error("PyTorch-FID accepts:")
        logger.error("  1. A directory of images")
        logger.error("  2. An NPZ file with precomputed statistics")
        return {'error': error_msg, 'method': 'pytorch_fid', 'invalid_type': True}
    
    # Count actual images
    actual_count = sum(
        1 for root, dirs, files in os.walk(images_dir)
        for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    
    logger.info(f"Generated images directory: {images_dir}")
    if is_npz_file:
        logger.info(f"Reference statistics file: {pytorch_fid_ref}")
    else:
        logger.info(f"Reference images directory: {pytorch_fid_ref}")
    logger.info(f"Generated images found: {actual_count}")
    
    # Build command
    cmd = [
        sys.executable,
        "-m", "pytorch_fid",
        images_dir,
        pytorch_fid_ref,
        "--device", "cuda"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info("")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.rstrip()
                logger.info(line)
                output_lines.append(line)
        
        process.wait()
        
        if process.returncode != 0:
            error_msg = f"PyTorch-FID calculation failed with return code {process.returncode}"
            logger.error(error_msg)
            return {'error': error_msg, 'method': 'pytorch_fid'}
        
        # Parse FID value from output (pytorch-fid outputs "FID:  <value>")
        fid_value = None
        for line in output_lines:
            if 'FID:' in line:
                try:
                    fid_value = float(line.split(':')[1].strip())
                    break
                except (ValueError, IndexError):
                    continue
        
        if fid_value is None:
            error_msg = "Could not parse FID value from pytorch-fid output"
            logger.error(error_msg)
            return {'error': error_msg, 'method': 'pytorch_fid'}
        
        results = {
            'method': 'pytorch_fid',
            'fid': fid_value,
            'num_images': actual_count,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("")
        logger.info("="*70)
        logger.info(f"PyTorch-FID Score: {fid_value:.4f}")
        logger.info("="*70)
        
        return results
    
    except Exception as e:
        error_msg = f"PyTorch-FID calculation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {'error': error_msg, 'method': 'pytorch_fid'}

