# FocusContourNet

FocusContourNet is a clean, standalone, self-owned detector project.
It is an end-to-end dual-head detector inspired by the YOLO26 design philosophy, while remaining a self-owned engineering line rather than an `ultralytics-main` modification.

Current focus areas:

- detect-only baseline training and validation
- pluggable Backbone / Neck focus-region enhancement
- parallel contour branch for future joint training
- detect-only, joint, and ablation-friendly experiments

This repository does not require the `ultralytics` Python package as a runtime dependency.

## Ubuntu Quick Start

Recommended installation flow:

```bash
git clone https://github.com/LL1125/focus-contour-net.git
cd focus-contour-net
conda create -n fcn python=3.10 -y
conda activate fcn
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
python scripts/check_model.py --model-config configs/model/fcn_base.yaml --img-size 320
```

A lightweight debug training command is also available:

```bash
python tools/train_detect.py --config configs/train/train_debug.yaml
```

## Dependency Policy

The repository keeps `requirements.txt` focused on direct project dependencies and contour-related utility needs.

- `torch`, `torchvision`, and `torchaudio` are installed separately to match CUDA 12.4 / cu124.
- ROS2, `ultralytics`, `nvidia-*`, `cuda-toolkit`, and unrelated environment packages are intentionally not part of this repository dependency list.
- Helper install or environment scripts, if added later, should remain optional convenience tools rather than the primary installation path.

## Current Status

- Detect-only baseline: available
- Joint detect + contour branch: scaffolded and ready for future formal module integration
- Focus plugin: lightweight replaceable placeholder
- Fourier contour head: executable interface shell for future formal implementation

## Ablation Targets

- `fcn_base`: baseline detect-only
- `fcn_focus_contour`: + focus plugin + contour branch
- `fcn_focus_contour_p2`: reserved small-object variant with P2 output flag

## Notes

- Paths use `pathlib` throughout for Ubuntu and Windows compatibility.
- The current contour branch is intentionally lightweight and marked for future replacement with the formal contour module.
- The engineering structure is designed to stay clean, pluggable, and easy to extend as the focus module and contour branch mature.
