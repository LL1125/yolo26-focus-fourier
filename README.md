# my_yolo26_fourier

A clean, standalone, self-owned YOLO26-style project for:

- detect-only baseline training and validation
- pluggable Backbone / Neck focus-region enhancement
- parallel Fourier contour head for future joint training
- detect-only, joint, and ablation-friendly experiments

This repository is a self-owned YOLO26-style engineering project. It is not an in-place modification of `ultralytics-main`, and it does not require the `ultralytics` Python package as a runtime dependency.

## Ubuntu Quick Start

Recommended installation flow:

```bash
git clone https://github.com/LL1125/yolo26-focus-fourier.git
cd yolo26-focus-fourier
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Why is `torch` installed separately?

- The project is expected to run against your Ubuntu CUDA 12.4 / cu124 environment.
- Keeping `torch`, `torchvision`, and `torchaudio` out of `requirements.txt` makes the repository easier to reuse across different CUDA or CPU setups.
- The recommended command above matches the validated environment target:
  - Python 3.10.20
  - torch 2.5.1+cu124
  - torchvision 0.20.1+cu124
  - torchaudio 2.5.1+cu124

## Sanity Check

After installation, you can run:

```bash
python scripts/check_model.py --model-config configs/model/y26_base.yaml --img-size 320
```

A lightweight debug training command is also available:

```bash
python tools/train_detect.py --config configs/train/train_debug.yaml
```

## Dependency Policy

The repository keeps `requirements.txt` focused on direct project dependencies and Fourier-related utility needs.

- `torch`, `torchvision`, and `torchaudio` are installed separately to match CUDA 12.4 / cu124.
- ROS2, `ultralytics`, `nvidia-*`, `cuda-toolkit`, and unrelated environment packages are intentionally not part of this repository dependency list.
- If helper install or environment scripts are added later, they should be treated as optional convenience tools rather than the primary installation path.

## Current Status

- Detect-only baseline: available
- Joint detect + contour branch: scaffolded and ready for future formal module integration
- Focus plugin: lightweight replaceable placeholder
- Fourier contour head: executable interface shell for future formal implementation

## Ablation Targets

- `y26_base`: baseline detect-only
- `y26_focus_fourier`: + focus plugin + contour branch
- `y26_focus_fourier_p2`: reserved small-object variant with P2 output flag

## Notes

- Paths use `pathlib` throughout for Ubuntu and Windows compatibility.
- The current contour branch is intentionally lightweight and marked for future replacement with the formal Fourier module.
- The engineering structure is designed to stay clean, pluggable, and easy to extend as the focus module and contour branch mature.
