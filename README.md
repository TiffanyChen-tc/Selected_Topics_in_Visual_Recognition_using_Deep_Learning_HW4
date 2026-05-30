# HW4 — Image Restoration with PromptIR

**Student ID:** 314553003  
**Name:** Yi-Chien Chen  

---

## Introduction

This repository contains my implementation for HW4 of *Selected Topics in Visual Recognition using Deep Learning*.  
The task is all-in-one image restoration: a single model must remove both rain streaks and snow particles from degraded images. Performance is measured by PSNR (Peak Signal-to-Noise Ratio) against the ground-truth clean images.

The method is based on **PromptIR** (Potlapalli et al., ECCV 2024), a transformer-based image restoration network that uses learnable prompt components to supply degradation-specific guidance at multiple decoder scales. All models are trained from scratch with no pretrained weights and no external data.

Key modifications over the vanilla PromptIR:

- **prompt_len 5 → 8** for the three original decoder prompt blocks, giving the model a richer basis to distinguish rain density, streak direction, and snow particle size.
- **prompt0**: an additional full-resolution prompt interaction block inserted between the Level-1 decoder and the refinement stage. It provides degradation-specific guidance at the highest spatial resolution, directly targeting fine structures such as rain streaks and snow particle edges.
- **8-fold D4 TTA** at inference time (4 rotations × identity/hflip), compatible with the training augmentation distribution.
- **np.rint rounding** when quantising float predictions to uint8, eliminating the systematic −0.5 LSB bias from plain floor truncation.

---

## Environment Setup

Recommended environment:

- Python 3.9 or higher
- PyTorch ≥ 2.0 with CUDA
- einops
- Pillow
- tqdm
- matplotlib

Install dependencies:

```bash
pip install torch torchvision einops pillow tqdm matplotlib
```

> **Windows note:** if you encounter TDR (GPU timeout) crashes, AMP (`--amp`, enabled by default) reduces per-kernel execution time and mitigates the issue.

---

## Usage

### Dataset layout

```text
hw4_realse_dataset/
├── train/
│   ├── degraded/
│   │   ├── rain-1.png … rain-1600.png
│   │   └── snow-1.png … snow-1600.png
│   └── clean/
│       ├── rain_clean-1.png … rain_clean-1600.png
│       └── snow_clean-1.png … snow_clean-1600.png
└── test/
    └── degraded/
        └── 0.png … 99.png
```

### Training

```bash
python train.py \
    --data_dir hw4_realse_dataset \
    --patch_size 192 \
    --epochs 200 \
    --batch_size 2 \
    --accum_steps 2 \
    --lr 2e-4 \
    --val_ratio 0.1 \
    --ckpt_dir checkpoints
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--patch_size` | 128 | Random crop size during training |
| `--epochs` | 150 | Total training epochs |
| `--batch_size` | 4 | Batch size per GPU |
| `--accum_steps` | 1 | Gradient accumulation steps |
| `--lr` | 2e-4 | Peak learning rate (cosine annealing after 15-epoch warmup) |
| `--val_ratio` | 0.1 | Fraction held out for validation (stratified by degradation type) |
| `--no_amp` | — | Disable mixed-precision training (AMP ON by default) |
| `--resume` | — | Path to checkpoint to resume from |


### Inference

```bash
python infer.py \
    --data_dir hw4_realse_dataset \
    --ckpt checkpoints/best_psnr.pth \
    --output results/pred.npz \
    --tile_size 2048 \
    --tile_overlap 128
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--tile_size` | 512 | Images larger than this are processed tile-by-tile |
| `--tile_overlap` | 64 | Overlap between adjacent tiles (pixels) |
| `--no_tta` | — | Disable 8-fold D4 TTA (TTA ON by default) |

---

## Performance Snapshot

Best public leaderboard result: **31.47 dB PSNR**

![Leaderboard Screenshot](leaderboard.png)

