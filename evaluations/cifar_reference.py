#!/usr/bin/env python
"""
Module for generating and managing CIFAR-10 FID reference files.

This module provides functionality to generate the NPZ reference file required
by the evaluator.py for computing evaluation metrics (IS, sFID, Precision, Recall).

The generated file contains the CIFAR-10 training dataset (50,000 images) in the
format expected by the guided-diffusion evaluator:
- Array name: 'arr_0'
- Shape: (50000, 32, 32, 3) - NHWC format
- dtype: uint8
- Value range: [0, 255]
"""

import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import Tuple, Optional


def load_cifar10_train_raw(data_root: str = "./data") -> np.ndarray:
    """
    Load CIFAR-10 training dataset in raw format (no normalization).
    
    Args:
        data_root: Root directory where CIFAR-10 data will be downloaded/stored
        
    Returns:
        NumPy array of shape (50000, 32, 32, 3) with uint8 values in [0, 255]
        Images are in NHWC (batch, height, width, channels) format
    """
    # Transform to convert PIL images to numpy arrays in [0, 255] range
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and CHW format
    ])
    
    print(f"Loading CIFAR-10 training set from {data_root}...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    print("Converting dataset to numpy array...")
    images_list = []
    
    for i in tqdm(range(len(trainset)), desc="Loading images"):
        img_tensor, _ = trainset[i]  # img_tensor is (C, H, W) in [0, 1]
        
        # Convert to NHWC format and scale to [0, 255]
        img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        images_list.append(img_np)
    
    # Stack all images into a single array
    images = np.stack(images_list, axis=0)
    
    print(f"Dataset shape: {images.shape}")
    print(f"Dataset dtype: {images.dtype}")
    print(f"Value range: [{images.min()}, {images.max()}]")
    
    return images


def save_cifar10_reference(
    images: np.ndarray,
    output_path: str,
    overwrite: bool = False
) -> None:
    """
    Save CIFAR-10 images to NPZ file in the format expected by evaluator.py.
    
    Args:
        images: NumPy array of shape (N, H, W, C) with uint8 values
        output_path: Path where the NPZ file will be saved
        overwrite: If True, overwrite existing file; if False, skip if file exists
        
    Raises:
        ValueError: If images don't have the expected shape or dtype
        FileExistsError: If file exists and overwrite=False
    """
    # Validate input
    if len(images.shape) != 4:
        raise ValueError(f"Expected 4D array (N, H, W, C), got shape {images.shape}")
    
    if images.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8, got {images.dtype}")
    
    if images.min() < 0 or images.max() > 255:
        raise ValueError(f"Values must be in [0, 255], got range [{images.min()}, {images.max()}]")
    
    # Check if file exists
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"File already exists: {output_path}\n"
            f"Set overwrite=True to replace it."
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as NPZ with key 'arr_0' (required by evaluator.py)
    print(f"Saving reference file to {output_path}...")
    np.savez(output_path, arr_0=images)
    
    # Verify the saved file
    print("Verifying saved file...")
    data = np.load(output_path)
    assert 'arr_0' in data.keys(), "Missing 'arr_0' key in saved file"
    assert data['arr_0'].shape == images.shape, "Shape mismatch in saved file"
    assert data['arr_0'].dtype == images.dtype, "Dtype mismatch in saved file"
    
    print(f"✓ Successfully saved {images.shape[0]} images to {output_path}")
    print(f"  Shape: {data['arr_0'].shape}")
    print(f"  Dtype: {data['arr_0'].dtype}")
    print(f"  Size: {os.path.getsize(output_path) / (1024**2):.2f} MB")


def generate_cifar10_reference(
    output_path: str,
    data_root: str = "./data",
    force_regenerate: bool = False
) -> str:
    """
    Generate CIFAR-10 FID reference file if it doesn't exist.
    
    This is the main function to use for generating the reference file.
    It checks if the file already exists and only generates it if needed.
    
    Args:
        output_path: Path where the NPZ file will be saved
        data_root: Root directory for downloading/storing CIFAR-10 data
        force_regenerate: If True, regenerate even if file exists
        
    Returns:
        Path to the reference file (same as output_path)
        
    Example:
        >>> from src.evaluations.cifar_reference import generate_cifar10_reference
        >>> ref_path = generate_cifar10_reference(
        ...     output_path="./fid-refs/cifar_dataset_50000.npz",
        ...     data_root="./data"
        ... )
        >>> print(f"Reference file ready at: {ref_path}")
    """
    # Check if file already exists
    if os.path.exists(output_path) and not force_regenerate:
        print(f"✓ Reference file already exists: {output_path}")
        
        # Verify the existing file
        try:
            data = np.load(output_path)
            assert 'arr_0' in data.keys(), "Missing 'arr_0' key"
            arr = data['arr_0']
            assert arr.shape == (50000, 32, 32, 3), f"Unexpected shape: {arr.shape}"
            assert arr.dtype == np.uint8, f"Unexpected dtype: {arr.dtype}"
            
            print(f"  Verified: shape={arr.shape}, dtype={arr.dtype}")
            print(f"  Value range: [{arr.min()}, {arr.max()}]")
            print("  Using existing file (no regeneration needed)")
            
            return output_path
            
        except Exception as e:
            print(f"⚠ Warning: Existing file validation failed: {e}")
            print("  Will regenerate the file...")
    
    # Generate the reference file
    print("\n" + "="*70)
    print("Generating CIFAR-10 FID Reference File")
    print("="*70)
    print()
    
    # Load CIFAR-10 training dataset
    images = load_cifar10_train_raw(data_root)
    
    # Save to NPZ file
    save_cifar10_reference(images, output_path, overwrite=force_regenerate)
    
    print()
    print("="*70)
    print("Reference file generation complete!")
    print("="*70)
    
    return output_path


def compare_reference_files(file1: str, file2: str) -> Tuple[bool, dict]:
    """
    Compare two NPZ reference files to check if they are identical.
    
    Args:
        file1: Path to first NPZ file
        file2: Path to second NPZ file
        
    Returns:
        Tuple of (are_identical: bool, comparison_report: dict)
        
    Example:
        >>> from src.evaluations.cifar_reference import compare_reference_files
        >>> identical, report = compare_reference_files(
        ...     "fid-refs/cifar_dataset_50000.npz",
        ...     "fid-refs/cifar_dataset_50000_new.npz"
        ... )
        >>> print(f"Files identical: {identical}")
        >>> print(f"Report: {report}")
    """
    report = {
        'file1': file1,
        'file2': file2,
        'file1_exists': os.path.exists(file1),
        'file2_exists': os.path.exists(file2),
    }
    
    # Check if files exist
    if not report['file1_exists']:
        report['error'] = f"File 1 does not exist: {file1}"
        return False, report
    
    if not report['file2_exists']:
        report['error'] = f"File 2 does not exist: {file2}"
        return False, report
    
    # Load both files
    print(f"Loading file 1: {file1}")
    data1 = np.load(file1)
    
    print(f"Loading file 2: {file2}")
    data2 = np.load(file2)
    
    # Compare keys
    keys1 = list(data1.keys())
    keys2 = list(data2.keys())
    
    report['keys1'] = keys1
    report['keys2'] = keys2
    report['keys_match'] = keys1 == keys2
    
    if not report['keys_match']:
        report['error'] = f"Keys don't match: {keys1} vs {keys2}"
        return False, report
    
    # Compare arrays for each key
    arrays_match = True
    array_reports = {}
    
    for key in keys1:
        arr1 = data1[key]
        arr2 = data2[key]
        
        arr_report = {
            'shape1': arr1.shape,
            'shape2': arr2.shape,
            'dtype1': str(arr1.dtype),
            'dtype2': str(arr2.dtype),
            'shape_match': arr1.shape == arr2.shape,
            'dtype_match': arr1.dtype == arr2.dtype,
        }
        
        if arr_report['shape_match'] and arr_report['dtype_match']:
            # Check if values are identical
            values_match = np.array_equal(arr1, arr2)
            arr_report['values_match'] = values_match
            
            if not values_match:
                # Compute differences
                diff = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
                arr_report['max_diff'] = float(diff.max())
                arr_report['mean_diff'] = float(diff.mean())
                arr_report['num_different'] = int(np.sum(arr1 != arr2))
                arrays_match = False
        else:
            arr_report['values_match'] = False
            arrays_match = False
        
        array_reports[key] = arr_report
    
    report['arrays'] = array_reports
    report['all_arrays_match'] = arrays_match
    
    return arrays_match, report


def print_comparison_report(report: dict) -> None:
    """
    Print a formatted comparison report.
    
    Args:
        report: Comparison report dictionary from compare_reference_files()
    """
    print("\n" + "="*70)
    print("CIFAR-10 Reference File Comparison Report")
    print("="*70)
    print()
    print(f"File 1: {report['file1']}")
    print(f"File 2: {report['file2']}")
    print()
    
    if 'error' in report:
        print(f"❌ Error: {report['error']}")
        return
    
    print(f"Keys match: {'✓' if report['keys_match'] else '✗'}")
    
    if not report['keys_match']:
        print(f"  File 1 keys: {report['keys1']}")
        print(f"  File 2 keys: {report['keys2']}")
        return
    
    print()
    print("Array Comparison:")
    print("-" * 70)
    
    for key, arr_report in report['arrays'].items():
        print(f"\nKey: '{key}'")
        print(f"  Shape: {arr_report['shape1']} vs {arr_report['shape2']} - {'✓' if arr_report['shape_match'] else '✗'}")
        print(f"  Dtype: {arr_report['dtype1']} vs {arr_report['dtype2']} - {'✓' if arr_report['dtype_match'] else '✗'}")
        
        if 'values_match' in arr_report:
            if arr_report['values_match']:
                print(f"  Values: ✓ Identical")
            else:
                print(f"  Values: ✗ Different")
                if 'max_diff' in arr_report:
                    print(f"    Max difference: {arr_report['max_diff']}")
                    print(f"    Mean difference: {arr_report['mean_diff']:.4f}")
                    print(f"    Different elements: {arr_report['num_different']}")
    
    print()
    print("="*70)
    
    if report['all_arrays_match']:
        print("✓ FILES ARE IDENTICAL")
    else:
        print("✗ FILES ARE DIFFERENT")
    
    print("="*70)


def main():
    """
    Command-line interface for generating and comparing CIFAR-10 reference files.
    
    Usage:
        # Generate reference file
        python -m src.datasets.cifar_reference generate --output ./fid-refs/cifar_dataset_50000.npz
        
        # Compare two reference files
        python -m src.datasets.cifar_reference compare --file1 ./fid-refs/cifar_dataset_50000.npz --file2 ./fid-refs/test.npz
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate and manage CIFAR-10 FID reference files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate CIFAR-10 reference file')
    generate_parser.add_argument(
        '--output',
        default='./fid-refs/cifar_dataset_50000.npz',
        help='Output NPZ file path (default: ./fid-refs/cifar_dataset_50000.npz)'
    )
    generate_parser.add_argument(
        '--data-root',
        default='./data',
        help='Root directory for CIFAR-10 data (default: ./data)'
    )
    generate_parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if file exists'
    )
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two reference files')
    compare_parser.add_argument(
        '--file1',
        required=True,
        help='First NPZ file to compare'
    )
    compare_parser.add_argument(
        '--file2',
        required=True,
        help='Second NPZ file to compare'
    )
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        output_path = generate_cifar10_reference(
            output_path=args.output,
            data_root=args.data_root,
            force_regenerate=args.force
        )
        print(f"\n✓ Reference file ready at: {output_path}")
        
    elif args.command == 'compare':
        identical, report = compare_reference_files(args.file1, args.file2)
        print_comparison_report(report)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

