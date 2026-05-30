"""
infer.py - Inference script for HW4 Image Restoration.

Runs a trained PromptIR checkpoint on the test degraded images and saves
the results as `pred.npz` (the required submission format).

pred.npz format:
    Keys  : original filenames, e.g. '0.png', '1.png', ...
    Values: numpy arrays of shape (3, H, W), dtype uint8, values 0-255.

Usage:
    python infer.py --data_dir hw4_realse_dataset --ckpt checkpoints/best_psnr.pth --output results\pred.npz --tile_size 2048 --tile_overlap 128
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset
from model import PromptIR


# ---------------------------------------------------------------------------
# Tile-based inference
# ---------------------------------------------------------------------------

def infer_with_tiling(model, img_tensor, tile_size: int = 512, tile_overlap: int = 64, device='cuda'):
    """
    Split a large image into overlapping tiles, infer each tile, and blend.
    Uses simple uniform averaging in the overlap regions.
    """
    _, _, H, W = img_tensor.shape
    stride = tile_size - tile_overlap

    h_starts = list(range(0, H - tile_size, stride)) + [max(0, H - tile_size)]
    w_starts = list(range(0, W - tile_size, stride)) + [max(0, W - tile_size)]

    output_sum = torch.zeros_like(img_tensor, dtype=torch.float32)
    weight_sum = torch.zeros(1, 1, H, W, dtype=torch.float32)

    amp = str(device) != 'cpu'
    with torch.no_grad():
        for h_s in h_starts:
            for w_s in w_starts:
                h_e = min(h_s + tile_size, H)
                w_e = min(w_s + tile_size, W)
                h_s = max(0, h_e - tile_size)
                w_s = max(0, w_e - tile_size)

                tile = img_tensor[:, :, h_s:h_e, w_s:w_e].to(device)
                with torch.amp.autocast(device_type='cuda', enabled=amp):
                    out = model(tile).float().cpu()

                output_sum[:, :, h_s:h_e, w_s:w_e] += out
                weight_sum[:, :, h_s:h_e, w_s:w_e] += 1.0

    return output_sum / weight_sum


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = PromptIR(decoder=True)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(device).eval()
    print(f'Loaded checkpoint: {args.ckpt}')

    test_dataset = TestDataset(data_dir=args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    amp_enabled = device.type == 'cuda'

    def run_model(x):
        """Run model on one tensor; fall back to tiling for large images."""
        _, _, H, W = x.shape
        if H <= args.tile_size and W <= args.tile_size:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_enabled):
                return model(x.to(device)).float().cpu()
        return infer_with_tiling(
            model, x,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            device=device,
        )

    results = {}

    for img_tensor, (filename,) in tqdm(test_loader, desc='Inferring'):
        if args.tta:
            # 8-fold D4 TTA: 4 rotations × (no flip / hflip)
            # Each (aug, deaug) pair is its own inverse.
            variants = [
                # identity
                (lambda x: x,
                 lambda x: x),
                # hflip
                (lambda x: torch.flip(x, dims=[3]),
                 lambda x: torch.flip(x, dims=[3])),
                # vflip
                (lambda x: torch.flip(x, dims=[2]),
                 lambda x: torch.flip(x, dims=[2])),
                # hflip + vflip  (= 180° rot)
                (lambda x: torch.flip(x, dims=[2, 3]),
                 lambda x: torch.flip(x, dims=[2, 3])),
                # rot90
                (lambda x: torch.rot90(x, k=1, dims=[2, 3]),
                 lambda x: torch.rot90(x, k=-1, dims=[2, 3])),
                # rot90 + hflip
                (lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3]),
                 lambda x: torch.rot90(torch.flip(x, dims=[3]), k=-1, dims=[2, 3])),
                # rot270
                (lambda x: torch.rot90(x, k=3, dims=[2, 3]),
                 lambda x: torch.rot90(x, k=-3, dims=[2, 3])),
                # rot270 + hflip
                (lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[3]),
                 lambda x: torch.rot90(torch.flip(x, dims=[3]), k=-3, dims=[2, 3])),
            ]

            preds = []
            for aug, deaug in variants:
                pred = run_model(aug(img_tensor))
                preds.append(deaug(pred))
            restored = torch.stack(preds, dim=0).mean(dim=0)
        else:
            restored = run_model(img_tensor)

        # Round to nearest integer (not floor) before quantising to uint8.
        # Reduces mean quantisation bias from -0.5 to ~0 LSB.
        restored = restored.squeeze(0).clamp(0.0, 1.0)
        restored_np = np.rint(restored.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        results[filename] = restored_np

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **results)
    print(f'\nSaved {len(results)} restored images → {args.output}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='PromptIR Inference for HW4')
    parser.add_argument('--data_dir',     type=str, default='hw4_realse_dataset')
    parser.add_argument('--ckpt',         type=str, required=True)
    parser.add_argument('--output',       type=str, default='pred.npz')
    parser.add_argument('--tile_size',    type=int, default=512,
                        help='Tile side length; images ≤ this are processed whole.')
    parser.add_argument('--tile_overlap', type=int, default=64,
                        help='Overlap between adjacent tiles (px).')
    parser.add_argument('--tta',    action='store_true', default=True,
                        help='8-fold D4 TTA (default ON).')
    parser.add_argument('--no_tta', dest='tta', action='store_false',
                        help='Disable TTA.')
    return parser.parse_args()


if __name__ == '__main__':
    infer(parse_args())