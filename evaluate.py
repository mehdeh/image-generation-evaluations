#!/usr/bin/env python
"""
Unified evaluation CLI for diffusion model outputs.

This module provides a command-line interface for evaluating generated images
using multiple evaluation methods. Run from project root:
  python evaluate.py <exp_dir> --fid [--ref=URL_or_path]
  python evaluate.py <exp_dir> --evaluator-pytorch
  python evaluate.py <exp_dir> --pytorch-fid [--ref=URL_or_path]
  python evaluate.py <exp_dir> --calculate-metrics [--ref=URL_or_path]

Reference files: check fid-refs/ and pretrain_models/ first; download if missing.
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Ensure project root is in path when run as script
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluations.cifar_reference import generate_cifar10_reference
from evaluations.cifar_calculate_metrics_ref import generate_cifar10_calculate_metrics_ref
from evaluations.pytorch_fid import run_pytorch_fid_evaluation as _run_pytorch_fid_evaluation
from evaluations.reference_downloader import download_reference_file, is_url

EVALUATION_OUTPUT_DIR = "evaluation"


def _evaluation_output_dir(exp_dir: str) -> str:
    """Return the evaluation output directory path, creating it if needed."""
    out_dir = os.path.join(exp_dir, EVALUATION_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def setup_logger(exp_dir: str, method_name: str) -> logging.Logger:
    """Set up logger for evaluation with both console and file output."""
    logger = logging.getLogger(f"evaluate_{method_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    out_dir = _evaluation_output_dir(exp_dir)
    log_file = os.path.join(out_dir, f"evaluation_{method_name}.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Log file: {log_file}")
    return logger


def _run_subprocess_module(module: str, args: list, cwd: str, logger: logging.Logger) -> tuple:
    """Run a Python module as subprocess, return (success, output_lines)."""
    cmd = [sys.executable, "-m", module] + args
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info("")
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + (os.pathsep + env.get("PYTHONPATH", ""))
    try:
        process = subprocess.Popen(
            cmd,
            cwd=_PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        for line in iter(process.stdout.readline, ""):
            if line:
                line = line.rstrip()
                logger.info(line)
                output_lines.append(line)
        process.wait()
        return process.returncode == 0, output_lines
    except Exception as e:
        logger.error(str(e), exc_info=True)
        return False, []


def run_fid_evaluation(
    exp_dir: str, config: Dict[str, Any], ref_override: Optional[str] = None
) -> Dict[str, Any]:
    """Run FID evaluation using fid.py."""
    logger = setup_logger(exp_dir, "fid")
    logger.info("=" * 70)
    logger.info("FID Evaluation (Custom Implementation)")
    logger.info("=" * 70)

    eval_config = config.get("evaluation", {})
    images_dir = os.path.join(exp_dir, "samples")
    fid_ref = ref_override or eval_config.get("fid_ref")
    num_expected = eval_config.get("fid_num_expected", 50000)

    if not os.path.exists(images_dir):
        return {"error": f"Images directory not found: {images_dir}", "method": "fid"}

    if not fid_ref:
        logger.error("FID reference not specified. Use --ref or configure fid_ref.")
        return {"error": "FID reference not specified", "method": "fid"}

    if not is_url(fid_ref) and not os.path.exists(fid_ref):
        logger.error(f"FID reference file not found: {fid_ref}")
        return {"error": f"Reference not found: {fid_ref}", "method": "fid"}

    actual_count = sum(
        1 for root, dirs, files in os.walk(images_dir)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    num_to_use = min(actual_count, num_expected)

    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Reference: {fid_ref}")
    logger.info(f"Images found: {actual_count}, using: {num_to_use}")

    success, output_lines = _run_subprocess_module(
        "evaluations.fid",
        ["calc", "--images", images_dir, "--ref", fid_ref, "--num", str(num_to_use)],
        _PROJECT_ROOT,
        logger,
    )

    if not success:
        return {"error": "FID calculation failed", "method": "fid"}

    fid_value = None
    for line in output_lines:
        line = line.strip()
        if line and line[0].isdigit():
            try:
                fid_value = float(line)
                break
            except ValueError:
                continue

    if fid_value is None:
        return {"error": "Could not parse FID value", "method": "fid"}

    results = {
        "method": "fid",
        "fid": fid_value,
        "num_images": num_to_use,
        "timestamp": datetime.now().isoformat(),
    }
    logger.info("")
    logger.info(f"FID Score: {fid_value:.4f}")
    return results


def run_evaluator_pytorch_evaluation(exp_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluation using evaluator_pytorch (IS, FID, sFID, Precision, Recall)."""
    logger = setup_logger(exp_dir, "evaluator_pytorch")
    logger.info("=" * 70)
    logger.info("Evaluator (PyTorch - No TensorFlow)")
    logger.info("=" * 70)

    eval_config = config.get("evaluation", {})
    images_npz = os.path.join(exp_dir, "allimages.npz")
    evaluator_ref = eval_config.get("evaluator_ref")

    if not os.path.exists(images_npz):
        return {"error": f"Images NPZ not found: {images_npz}", "method": "evaluator_pytorch"}

    if not evaluator_ref:
        return {"error": "evaluator_ref not configured", "method": "evaluator_pytorch"}

    if not os.path.exists(evaluator_ref):
        logger.info("Reference not found. Generating CIFAR-10 reference...")
        try:
            evaluator_ref = generate_cifar10_reference(
                output_path=evaluator_ref,
                data_root=os.path.join(_PROJECT_ROOT, "data"),
                force_regenerate=False,
            )
        except Exception as e:
            return {"error": str(e), "method": "evaluator_pytorch"}

    success, output_lines = _run_subprocess_module(
        "evaluations.evaluator_pytorch",
        [evaluator_ref, images_npz],
        _PROJECT_ROOT,
        logger,
    )

    if not success:
        return {"error": "Evaluator failed", "method": "evaluator_pytorch"}

    metrics = {"method": "evaluator_pytorch"}
    for line in output_lines:
        if "Inception Score:" in line:
            metrics["inception_score"] = float(line.split(":")[1].strip())
        elif "sFID:" in line:
            metrics["sfid"] = float(line.split(":")[1].strip())
        elif "FID:" in line and "sFID" not in line:
            metrics["fid"] = float(line.split(":")[1].strip())
        elif "Precision:" in line:
            metrics["precision"] = float(line.split(":")[1].strip())
        elif "Recall:" in line:
            metrics["recall"] = float(line.split(":")[1].strip())
    metrics["timestamp"] = datetime.now().isoformat()
    return metrics


def run_evaluator_evaluation(exp_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run TensorFlow-based evaluator (requires TensorFlow)."""
    logger = setup_logger(exp_dir, "evaluator")
    logger.info("=" * 70)
    logger.info("Evaluator (TensorFlow - guided-diffusion)")
    logger.info("=" * 70)

    eval_config = config.get("evaluation", {})
    images_npz = os.path.join(exp_dir, "allimages.npz")
    evaluator_ref = eval_config.get("evaluator_ref")

    if not os.path.exists(images_npz):
        return {"error": f"Images NPZ not found: {images_npz}", "method": "evaluator"}

    if not evaluator_ref:
        return {"error": "evaluator_ref not configured", "method": "evaluator"}

    if not os.path.exists(evaluator_ref):
        logger.info("Generating CIFAR-10 reference...")
        try:
            evaluator_ref = generate_cifar10_reference(
                output_path=evaluator_ref,
                data_root=os.path.join(_PROJECT_ROOT, "data"),
                force_regenerate=False,
            )
        except Exception as e:
            return {"error": str(e), "method": "evaluator"}

    success, output_lines = _run_subprocess_module(
        "evaluations.evaluator",
        [evaluator_ref, images_npz],
        _PROJECT_ROOT,
        logger,
    )

    if not success:
        return {"error": "Evaluator failed", "method": "evaluator"}

    metrics = {"method": "evaluator"}
    for line in output_lines:
        if "Inception Score:" in line:
            metrics["inception_score"] = float(line.split(":")[1].strip())
        elif "sFID:" in line:
            metrics["sfid"] = float(line.split(":")[1].strip())
        elif "FID:" in line and "sFID" not in line:
            metrics["fid"] = float(line.split(":")[1].strip())
        elif "Precision:" in line:
            metrics["precision"] = float(line.split(":")[1].strip())
        elif "Recall:" in line:
            metrics["recall"] = float(line.split(":")[1].strip())
    metrics["timestamp"] = datetime.now().isoformat()
    return metrics


def run_calculate_metrics_evaluation(
    exp_dir: str, config: Dict[str, Any], ref_override: Optional[str] = None
) -> Dict[str, Any]:
    """Run FID and FD_DINOv2 evaluation using calculate_metrics."""
    logger = setup_logger(exp_dir, "calculate_metrics")
    logger.info("=" * 70)
    logger.info("Calculate Metrics (FID + FD_DINOv2)")
    logger.info("=" * 70)

    eval_config = config.get("evaluation", {})
    images_dir = os.path.join(exp_dir, "samples")
    num_expected = eval_config.get("fid_num_expected", 50000)

    ref_path = ref_override
    if not ref_path:
        full_ref = eval_config.get("calculate_metrics_ref_full")
        if full_ref and not is_url(full_ref):
            if os.path.exists(full_ref):
                ref_path = full_ref
            else:
                logger.info("Generating full reference (FID+FD_DINOv2)...")
                try:
                    ref_path = generate_cifar10_calculate_metrics_ref(
                        output_path=full_ref,
                        data_root=os.path.join(_PROJECT_ROOT, "data"),
                        force_regenerate=False,
                    )
                except Exception as e:
                    logger.warning(f"Full ref failed: {e}, falling back to npz")
        if not ref_path:
            ref_path = eval_config.get("calculate_metrics_ref")

    if not os.path.exists(images_dir):
        return {"error": f"Images directory not found: {images_dir}", "method": "calculate_metrics"}

    if not ref_path:
        return {"error": "Reference not specified", "method": "calculate_metrics"}

    if not is_url(ref_path) and not os.path.exists(ref_path):
        return {"error": f"Reference not found: {ref_path}", "method": "calculate_metrics"}

    actual_count = sum(
        1 for root, dirs, files in os.walk(images_dir)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    num_to_use = min(actual_count, num_expected)

    success, output_lines = _run_subprocess_module(
        "evaluations.calculate_metrics",
        ["calc", "--images", images_dir, "--ref", ref_path, "--num", str(num_to_use), "--json"],
        _PROJECT_ROOT,
        logger,
    )

    if not success:
        return {"error": "Calculate metrics failed", "method": "calculate_metrics"}

    json_str = "\n".join(output_lines)
    json_start = json_str.find("{")
    if json_start >= 0:
        json_str = json_str[json_start:]
    try:
        results = json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON output", "method": "calculate_metrics"}
    results["timestamp"] = datetime.now().isoformat()
    return results


def run_pytorch_fid_evaluation(
    exp_dir: str, config: Dict[str, Any], ref_override: Optional[str] = None
) -> Dict[str, Any]:
    """Run FID using official pytorch-fid library."""
    logger = setup_logger(exp_dir, "pytorch_fid")

    if ref_override:
        if is_url(ref_override):
            try:
                ref_override = download_reference_file(
                    ref_override,
                    cache_dir=os.path.join(_PROJECT_ROOT, "fid-refs"),
                    verbose=True,
                )
            except Exception as e:
                return {"error": str(e), "method": "pytorch_fid"}
        config = dict(config)
        config.setdefault("evaluation", {})["pytorch_fid_ref"] = ref_override
    else:
        eval_config = config.get("evaluation", {})
        pytorch_fid_ref = eval_config.get("pytorch_fid_ref")
        if pytorch_fid_ref and is_url(pytorch_fid_ref):
            try:
                pytorch_fid_ref = download_reference_file(
                    pytorch_fid_ref,
                    cache_dir=os.path.join(_PROJECT_ROOT, "fid-refs"),
                    verbose=True,
                )
                config = dict(config)
                config.setdefault("evaluation", {})["pytorch_fid_ref"] = pytorch_fid_ref
            except Exception as e:
                return {"error": str(e), "method": "pytorch_fid"}

    return _run_pytorch_fid_evaluation(exp_dir, config, logger)


def save_results(exp_dir: str, results: Dict[str, Any], method: str) -> None:
    """Save evaluation results to JSON."""
    out_dir = _evaluation_output_dir(exp_dir)
    results_file = os.path.join(out_dir, f"evaluation_{method}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated images using different methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  --fid               Custom FID (fid.py)
  --evaluator         TensorFlow evaluator (IS, sFID, Precision, Recall)
  --evaluator-pytorch PyTorch evaluator (no TensorFlow)
  --pytorch-fid       Official pytorch-fid
  --calculate-metrics FID + FD_DINOv2

Examples:
  python evaluate.py experiments/exp_001/ --fid --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
  python evaluate.py experiments/exp_001/ --evaluator-pytorch
  python evaluate.py experiments/exp_001/ --calculate-metrics
        """,
    )
    parser.add_argument("exp_dir", help="Path to experiment directory (samples/ or allimages.npz)")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--ref", help="Reference path or URL (overrides config)")
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--fid", action="store_true", help="Run FID")
    method_group.add_argument("--evaluator", action="store_true", help="Run TensorFlow evaluator")
    method_group.add_argument("--evaluator-pytorch", action="store_true", help="Run PyTorch evaluator")
    method_group.add_argument("--pytorch-fid", action="store_true", help="Run pytorch-fid")
    method_group.add_argument("--calculate-metrics", action="store_true", help="Run FID + FD_DINOv2")

    args = parser.parse_args()

    if not os.path.exists(args.exp_dir):
        print(f"Error: Experiment directory not found: {args.exp_dir}")
        sys.exit(1)

    from src.utils.config import create_default_config, load_config, merge_configs

    default_config = create_default_config()
    if args.config and os.path.exists(args.config):
        user_config = load_config(args.config)
        config = merge_configs(default_config, user_config)
    else:
        exp_config_path = os.path.join(args.exp_dir, "config.yaml")
        if os.path.exists(exp_config_path):
            config = merge_configs(default_config, load_config(exp_config_path))
        else:
            config = default_config

    eval_config = config.get("evaluation", {})
    default_eval = default_config.get("evaluation", {})
    for key in ["fid_ref", "pytorch_fid_ref", "evaluator_ref", "calculate_metrics_ref", "calculate_metrics_ref_full"]:
        val = eval_config.get(key)
        default_val = default_eval.get(key)
        if val and default_val and not is_url(val) and not os.path.exists(val) and is_url(default_val):
            eval_config[key] = default_val

    results = None
    method_name = None

    if args.fid:
        method_name = "fid"
        results = run_fid_evaluation(args.exp_dir, config, args.ref)
    elif args.evaluator:
        method_name = "evaluator"
        results = run_evaluator_evaluation(args.exp_dir, config)
    elif args.evaluator_pytorch:
        method_name = "evaluator_pytorch"
        results = run_evaluator_pytorch_evaluation(args.exp_dir, config)
    elif args.pytorch_fid:
        method_name = "pytorch_fid"
        results = run_pytorch_fid_evaluation(args.exp_dir, config, args.ref)
    elif args.calculate_metrics:
        method_name = "calculate_metrics"
        results = run_calculate_metrics_evaluation(args.exp_dir, config, args.ref)

    if results:
        save_results(args.exp_dir, results, method_name)
        if "error" in results:
            sys.exit(1)
    else:
        print("Error: No results")
        sys.exit(1)


if __name__ == "__main__":
    main()
