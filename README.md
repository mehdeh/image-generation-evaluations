# Image Generation Evaluations

A unified toolkit for evaluating generated images from diffusion models and other generative models. Provides multiple evaluation metrics with automatic download of reference files and pretrained models.

## Features

- **Multiple evaluation methods**: FID, Inception Score, sFID, Precision, Recall, FD_DINOv2
- **Auto-download**: Reference files (`fid-refs/`) and pretrained models (`pretrain_models/`) are checked first and downloaded if missing
- **Standalone**: Self-contained project with minimal dependencies
- **Modular**: Clean separation of evaluation methods

## Installation

```bash
cd images_generation_evaluations
pip install -r requirements.txt
```

## Quick Start

```bash
# FID evaluation (samples in exp_dir/samples/)
python evaluate.py experiments/exp_001/ --fid --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

# PyTorch evaluator (IS, FID, sFID, Precision, Recall) - requires allimages.npz
python evaluate.py experiments/exp_001/ --evaluator-pytorch

# FID + FD_DINOv2
python evaluate.py experiments/exp_001/ --calculate-metrics
```

## Evaluation Methods

### 1. FID (Fréchet Inception Distance) — `fid.py`

Measures the similarity between generated and real image distributions using Inception-v3 features. Lower is better.

- **Paper**: [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) (Heusel et al., NeurIPS 2017)
- **Original implementation**: [bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)
- **Input**: `exp_dir/samples/` (PNG/JPG images)
- **Reference**: NPZ file with precomputed (mu, sigma) or URL

### 2. Evaluator (TensorFlow) — `evaluator.py`

Computes Inception Score, FID, sFID (spatial FID), Precision, and Recall. Uses TensorFlow and Inception graph.

- **Source**: [openai/guided-diffusion](https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py)
- **Paper**: [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011)
- **Input**: `exp_dir/allimages.npz` (NHWC uint8 array)
- **Reference**: NPZ with `arr_0` (reference images)

### 3. Evaluator (PyTorch) — `evaluator_pytorch.py`

Same metrics as the TensorFlow evaluator but implemented in PyTorch. No TensorFlow dependency.

- **Derived from**: [evaluator.py](https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py) (converted to PyTorch)
- **Precision/Recall**: [kynkaat/improved-precision-and-recall-metric](https://github.com/kynkaat/improved-precision-and-recall-metric) (NeurIPS 2019)
- **Metrics**: Inception Score, FID, sFID, Precision, Recall
- **Input**: `exp_dir/allimages.npz`
- **Reference**: NPZ with `arr_0` (e.g. CIFAR-10 training set)

### 4. PyTorch-FID — `pytorch_fid.py`

Wrapper around the official [pytorch-fid](https://github.com/mseitzer/pytorch-fid) library.

- **GitHub**: [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- **Input**: `exp_dir/samples/` and reference (directory or NPZ)
- **Install**: `pip install pytorch-fid`

### 5. Calculate Metrics (FID + FD_DINOv2) — `calculate_metrics.py`

Computes FID (Inception-based) and FD_DINOv2 (DINOv2-based Fréchet Distance).

- **Source**: [NVlabs/edm2](https://github.com/NVlabs/edm2/blob/main/calculate_metrics.py)
- **FD_DINOv2**: Uses DINOv2 features for modern evaluation
- **Input**: `exp_dir/samples/`
- **Reference**: NPZ (FID only) or PKL (FID + FD_DINOv2)

## Directory Structure

```
images_generation_evaluations/
├── evaluate.py           # Main CLI entry point
├── config/
│   └── default.yaml      # Default evaluation config
├── evaluations/          # Evaluation method implementations
│   ├── fid.py
│   ├── evaluator.py
│   ├── evaluator_pytorch.py
│   ├── calculate_metrics.py
│   ├── pytorch_fid.py
│   ├── cifar_reference.py
│   ├── cifar_calculate_metrics_ref.py
│   └── reference_downloader.py
├── fid-refs/             # FID reference statistics (auto-downloaded)
├── pretrain_models/      # Inception, etc. (auto-downloaded)
├── src/
│   ├── datasets/
│   └── utils/
├── dnnlib/
├── torch_utils/
└── README.md
```

## Configuration

Edit `config/default.yaml` or pass `--config path/to/config.yaml`. Reference paths can be local or URLs; URLs are downloaded to `fid-refs/` when needed.

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, Pillow, tqdm, click, PyYAML
- Optional: TensorFlow (for `evaluator.py`), pytorch-fid (for `--pytorch-fid`)

## License and Attribution

**This project's own code** is provided as-is for free use. It has no formal license; you may use, modify, and distribute it freely.

**Third-party code** included or adapted from other repositories retains its original license:

- **evaluator.py**: From [openai/guided-diffusion](https://github.com/openai/guided-diffusion) (MIT)
- **calculate_metrics.py**: From [NVlabs/edm2](https://github.com/NVlabs/edm2) (CC BY-NC-SA 4.0)
- **fid.py**, **dnnlib**, **torch_utils**: From NVIDIA research (CC BY-NC-SA 4.0)
- **pytorch-fid**: Apache 2.0 when used as external dependency

Please comply with the respective licenses of any code you use or redistribute.

---

## Suggested Repository Name

**`gen-image-eval`** or **`image-generation-evaluations`**

## Suggested GitHub Description

> Unified toolkit for evaluating generated images: FID, Inception Score, sFID, Precision, Recall, FD_DINOv2. Auto-downloads references and models. PyTorch-first with optional TensorFlow support.
