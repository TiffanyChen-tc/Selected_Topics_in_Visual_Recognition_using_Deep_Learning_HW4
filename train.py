"""
train.py - Training script for HW4 Image Restoration (Rain & Snow) using PromptIR.

Outputs saved to ``--ckpt_dir``:
    - promptir_epoch<N>.pth        Model checkpoints.
    - loss_curve.png               Train / val L1-loss per epoch.
    - psnr_per_type.png            Per-degradation-type PSNR bar chart (final epoch).
    - training_history.csv         Numeric log of every epoch.

Usage:
    python train.py
    python train.py --patch_size 160 --epochs 180 --batch_size 4 --accum_steps 1
    python train.py --patch_size 192 --epochs 200 --batch_size 2 --accum_steps 2
    python train.py --patch_size 208 --epochs 200 --batch_size 2 --accum_steps 2 --ckpt_dir checkpoints --resume checkpoints/last.pth
    python train.py --patch_size 192 --epochs 200 --batch_size 2 --accum_steps 2 --ckpt_dir checkpoints --resume checkpoints/last.pth
"""

import argparse
import csv
import math
import os
import random

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for servers / Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ValDataset, TrainDataset, build_pairs
from model import PromptIR


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute the mean PSNR (dB) over a batch.

    Both tensors are expected to be in [0, 1] with shape (B, C, H, W).
    Returns a Python float.
    """
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse.item())


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup followed by cosine annealing down to 0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / val split (stratified by degradation type)
# ---------------------------------------------------------------------------

def stratified_split(pairs: list, val_ratio: float, seed: int = 42):
    """
    Split *pairs* into (train_pairs, val_pairs) while preserving the
    rain / snow ratio in both subsets.

    Args:
        pairs:     Full list of pair dicts from :func:`build_pairs`.
        val_ratio: Fraction of each type to hold out for validation.
        seed:      Random seed for reproducibility.

    Returns:
        train_pairs, val_pairs
    """
    rng = random.Random(seed)

    rain = [p for p in pairs if p['deg_type'] == 'rain']
    snow = [p for p in pairs if p['deg_type'] == 'snow']

    rng.shuffle(rain)
    rng.shuffle(snow)

    n_rain_val = max(1, int(len(rain) * val_ratio))
    n_snow_val = max(1, int(len(snow) * val_ratio))

    val_pairs = rain[:n_rain_val] + snow[:n_snow_val]
    train_pairs = rain[n_rain_val:] + snow[n_snow_val:]

    rng.shuffle(train_pairs)
    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, criterion, device, amp_enabled=False):
    """
    Run one validation pass.

    Returns:
        val_loss  (float): Mean L1 loss across all batches.
        psnr_rain (float): Mean PSNR on rain images.
        psnr_snow (float): Mean PSNR on snow images.
        psnr_all  (float): Mean PSNR across all images.
    """
    model.eval()

    total_loss = 0.0
    psnr_by_type = {'rain': [], 'snow': []}

    for degraded, clean, deg_types in val_loader:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        # AMP forward (same as training); cast back to fp32 before computing metrics
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            restored = model(degraded)

        restored_fp32 = restored.float()
        clean_fp32 = clean.float()

        loss = criterion(restored_fp32, clean_fp32)
        total_loss += loss.item()

        # Per-sample PSNR (batch size is 1 during validation)
        psnr = batch_psnr(restored_fp32, clean_fp32)
        # deg_types is a tuple/list of strings from the DataLoader
        for dtype in deg_types:
            psnr_by_type[dtype].append(psnr)

    avg_loss = total_loss / len(val_loader)
    psnr_rain = sum(psnr_by_type['rain']) / len(psnr_by_type['rain']) if psnr_by_type['rain'] else 0.0
    psnr_snow = sum(psnr_by_type['snow']) / len(psnr_by_type['snow']) if psnr_by_type['snow'] else 0.0
    psnr_all = sum(psnr_by_type['rain'] + psnr_by_type['snow']) / (
        len(psnr_by_type['rain']) + len(psnr_by_type['snow'])
    )

    return avg_loss, psnr_rain, psnr_snow, psnr_all


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_loss_curve(train_losses, val_losses, out_path: str):
    """
    Save a train / val L1-loss curve to *out_path*.

    The x-axis is epoch number (1-indexed).  Both curves are plotted on the
    same axes.  The best val-loss epoch is highlighted with a vertical dashed
    line and annotated.
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label='Train L1 Loss', color='steelblue', linewidth=2)
    ax.plot(epochs, val_losses,   label='Val L1 Loss',   color='tomato',    linewidth=2)

    # Mark the best val epoch
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val = min(val_losses)
    ax.axvline(best_epoch, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.annotate(
        f'Best val\nEpoch {best_epoch}\n{best_val:.4f}',
        xy=(best_epoch, best_val),
        xytext=(best_epoch + max(1, len(epochs) * 0.03), best_val),
        fontsize=8,
        color='gray',
        arrowprops=dict(arrowstyle='->', color='gray'),
    )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('L1 Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curve saved to {out_path}')


def save_psnr_bar_chart(
    psnr_rain_hist,
    psnr_snow_hist,
    psnr_all_hist,
    out_path: str,
):
    """
    Save a per-degradation-type PSNR bar chart and an overlay line chart
    showing PSNR progression across epochs.

    Layout: two subplots side by side.
      Left  — PSNR vs Epoch (line chart, one curve per type + overall).
      Right — Final-epoch PSNR bar chart (rain / snow / overall).

    This is the meaningful equivalent of a "confusion matrix" for a
    regression-based restoration task: it shows how well the model handles
    each degradation type, analogous to a per-class performance breakdown.
    """
    epochs = list(range(1, len(psnr_rain_hist) + 1))

    fig, (ax_line, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: PSNR over epochs ----
    ax_line.plot(epochs, psnr_rain_hist, label='Rain',    color='royalblue',  linewidth=2)
    ax_line.plot(epochs, psnr_snow_hist, label='Snow',    color='mediumorchid', linewidth=2)
    ax_line.plot(epochs, psnr_all_hist,  label='Overall', color='darkorange',  linewidth=2, linestyle='--')
    ax_line.set_xlabel('Epoch', fontsize=12)
    ax_line.set_ylabel('PSNR (dB)', fontsize=12)
    ax_line.set_title('Val PSNR per Degradation Type', fontsize=13)
    ax_line.legend(fontsize=11)
    ax_line.grid(True, linestyle='--', alpha=0.4)

    # ---- Right: bar chart at the final epoch ----
    categories = ['Rain', 'Snow', 'Overall']
    values = [psnr_rain_hist[-1], psnr_snow_hist[-1], psnr_all_hist[-1]]
    colors = ['royalblue', 'mediumorchid', 'darkorange']

    bars = ax_bar.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{val:.2f} dB',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
        )

    # Horizontal reference lines
    for ref_db in [25, 30, 35]:
        ax_bar.axhline(ref_db, color='gray', linestyle=':', linewidth=0.9, alpha=0.6)
        ax_bar.text(len(categories) - 0.45, ref_db + 0.05, f'{ref_db} dB', fontsize=8, color='gray')

    ax_bar.set_ylim(0, max(values) + 3)
    ax_bar.set_ylabel('PSNR (dB)', fontsize=12)
    ax_bar.set_title(f'Final Epoch PSNR by Degradation Type\n'
                     f'(epoch {epochs[-1]})', fontsize=13)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.4)

    # Legend patches that match the line plot colours
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, categories)]
    ax_bar.legend(handles=patches, fontsize=10)

    fig.suptitle('Per-Type PSNR Analysis  ·  PromptIR HW4', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'PSNR bar chart saved to {out_path}')


def save_csv_log(history: list, out_path: str):
    """Append one row per epoch to a CSV for later analysis."""
    write_header = not os.path.exists(out_path)
    with open(out_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(history)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ---- Build pair list and stratified split ----
    all_pairs = build_pairs(args.data_dir)
    train_pairs, val_pairs = stratified_split(all_pairs, val_ratio=args.val_ratio)

    train_dataset = TrainDataset(train_pairs, patch_size=args.patch_size)
    val_dataset = ValDataset(val_pairs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # Val: batch_size=1 because images differ in resolution
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    model = PromptIR(decoder=True).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params / 1e6:.2f} M')

    # ---- Loss / optimiser / scheduler ----
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, warmup_epochs=15, total_epochs=args.epochs)

    # ---- AMP GradScaler (fp16 training — faster kernels, lower VRAM, helps avoid TDR) ----
    amp_enabled = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    print(f'AMP (mixed precision): {"ON" if amp_enabled else "OFF"}')

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- Optionally resume ----
    start_epoch = 0
    best_val_loss = float('inf')
    best_psnr_all = 0.0
    train_losses, val_losses = [], []
    psnr_rain_hist, psnr_snow_hist, psnr_all_hist = [], [], []
    csv_history = []

    if args.resume and os.path.isfile(args.resume):
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        # Restore history if saved
        train_losses = ckpt.get('train_losses', [])
        val_losses = ckpt.get('val_losses', [])
        psnr_rain_hist = ckpt.get('psnr_rain_hist', [])
        psnr_snow_hist = ckpt.get('psnr_snow_hist', [])
        psnr_all_hist = ckpt.get('psnr_all_hist', [])
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_psnr_all = ckpt.get('best_psnr_all', 0.0)
        scaler_state = ckpt.get('scaler_state_dict')
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        print(f'Resumed at epoch {start_epoch}')
    elif args.resume:
        print(f'Warning: checkpoint not found at {args.resume}, training from scratch.')

    # ---- Epoch loop ----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]

        pbar = tqdm(
            train_loader,
            desc=f'Epoch [{epoch + 1}/{args.epochs}]  lr={current_lr:.2e}',
            dynamic_ncols=True,
        )
        optimizer.zero_grad(set_to_none=True)
        for step, (degraded, clean) in enumerate(pbar):
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            # AMP autocast: fp16 forward pass — faster kernel execution reduces TDR risk
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                restored = model(degraded)
                # Scale loss for gradient accumulation
                loss = criterion(restored, clean) / args.accum_steps

            # scaler.scale handles fp16 gradient scaling automatically
            scaler.scale(loss).backward()

            # Only step the optimiser every accum_steps batches (or at end of epoch)
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                # Unscale before clipping so the norm is in fp32 scale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * args.accum_steps  # restore original scale for logging
            pbar.set_postfix({
                'loss': f'{loss.item() * args.accum_steps:.4f}',
                'accum': f'{(step % args.accum_steps) + 1}/{args.accum_steps}',
            })

        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- Validation ----
        val_loss, psnr_rain, psnr_snow, psnr_all = validate(
            model, val_loader, criterion, device, amp_enabled
        )
        val_losses.append(val_loss)
        psnr_rain_hist.append(psnr_rain)
        psnr_snow_hist.append(psnr_snow)
        psnr_all_hist.append(psnr_all)

        print(
            f'Epoch [{epoch + 1}/{args.epochs}] '
            f'| train_loss={avg_train_loss:.4f} '
            f'| val_loss={val_loss:.4f} '
            f'| PSNR rain={psnr_rain:.2f} dB  snow={psnr_snow:.2f} dB  all={psnr_all:.2f} dB'
        )

        csv_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'psnr_rain': psnr_rain,
            'psnr_snow': psnr_snow,
            'psnr_all': psnr_all,
            'lr': current_lr,
        })

        # ---- Plots (every epoch) ----
        save_loss_curve(
            train_losses, val_losses,
            os.path.join(args.ckpt_dir, 'loss_curve.png'),
        )
        save_psnr_bar_chart(
            psnr_rain_hist, psnr_snow_hist, psnr_all_hist,
            os.path.join(args.ckpt_dir, 'psnr_per_type.png'),
        )

        # ---- Shared checkpoint payload ----
        ckpt_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'best_psnr_all': best_psnr_all,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'psnr_rain_hist': psnr_rain_hist,
            'psnr_snow_hist': psnr_snow_hist,
            'psnr_all_hist': psnr_all_hist,
            'scaler_state_dict': scaler.state_dict(),
        }

        # ---- Save last.pth (always) ----
        torch.save(ckpt_payload, os.path.join(args.ckpt_dir, 'last.pth'))

        # ---- Save best_loss.pth (lowest val loss) ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_payload['best_val_loss'] = best_val_loss
            torch.save(ckpt_payload, os.path.join(args.ckpt_dir, 'best_loss.pth'))
            print(f'  ↑ New best val_loss={best_val_loss:.4f} — best_loss.pth updated')

        # ---- Save best_psnr.pth (highest val PSNR — what the leaderboard measures) ----
        if psnr_all > best_psnr_all:
            best_psnr_all = psnr_all
            ckpt_payload['best_psnr_all'] = best_psnr_all
            torch.save(ckpt_payload, os.path.join(args.ckpt_dir, 'best_psnr.pth'))
            print(f'  ↑ New best PSNR={best_psnr_all:.2f} dB — best_psnr.pth updated')

    # ---- Final CSV log ----
    csv_path = os.path.join(args.ckpt_dir, 'training_history.csv')
    save_csv_log(csv_history, csv_path)
    print(f'Training history saved → {csv_path}')
    print('Training complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train PromptIR for HW4 Image Restoration')

    # Paths
    parser.add_argument('--data_dir', type=str, default='hw4_realse_dataset',
                        help='Root directory of hw4_realse_dataset (contains train/ and test/).')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints and plots.')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to a checkpoint to resume training from.')

    # Training hyper-parameters
    parser.add_argument('--epochs', type=int, default=150,
                        help='Total number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Initial learning rate.')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Random crop patch size for training.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes.')

    # Validation / checkpointing
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of each degradation type held out for validation '
                             '(stratified split). Default: 0.1 → 10%%.')
    parser.add_argument('--no_amp', dest='amp', action='store_false',
                        help='Disable AMP and train in full fp32 (default: AMP ON).')
    parser.set_defaults(amp=True)
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = batch_size x accum_steps.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)