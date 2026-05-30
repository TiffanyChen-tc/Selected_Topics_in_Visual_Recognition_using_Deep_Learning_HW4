"""
dataset.py - Dataset classes for HW4 Image Restoration (Rain & Snow).

Training dataset structure:
    train/
        degraded/
            rain-1.png ... rain-1600.png
            snow-1.png ... snow-1600.png
        clean/
            rain_clean-1.png ... rain_clean-1600.png
            snow_clean-1.png ... snow_clean-1600.png

Test dataset structure:
    test/
        degraded/
            0.png ... 99.png
"""

import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def random_augment(degraded, clean):
    """Apply the same random flip / rotation to a degraded-clean pair."""
    if random.random() > 0.5:
        degraded = TF.hflip(degraded)
        clean = TF.hflip(clean)
    if random.random() > 0.5:
        degraded = TF.vflip(degraded)
        clean = TF.vflip(clean)
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        degraded = TF.rotate(degraded, angle)
        clean = TF.rotate(clean, angle)
    return degraded, clean


def crop_to_multiple(img, base=16):
    """Crop a PIL image so that H and W are both divisible by `base`."""
    w, h = img.size
    return img.crop((0, 0, w - w % base, h - h % base))


def build_pairs(data_dir: str) -> list:
    """
    Scan ``train/degraded/`` and return a list of dicts::

        {
            'degraded': <absolute path to degraded image>,
            'clean':    <absolute path to clean image>,
            'deg_type': 'rain' | 'snow',
        }

    This is kept separate from the Dataset constructors so the caller can
    perform a train / val split before building the actual datasets.
    """
    degraded_dir = os.path.join(data_dir, 'train', 'degraded')
    clean_dir = os.path.join(data_dir, 'train', 'clean')

    pairs = []
    for fname in sorted(os.listdir(degraded_dir)):
        if not fname.lower().endswith('.png'):
            continue

        if fname.startswith('rain-'):
            deg_type = 'rain'
            clean_fname = fname.replace('rain-', 'rain_clean-')
        elif fname.startswith('snow-'):
            deg_type = 'snow'
            clean_fname = fname.replace('snow-', 'snow_clean-')
        else:
            continue  # skip unexpected filenames

        clean_path = os.path.join(clean_dir, clean_fname)
        if not os.path.exists(clean_path):
            raise FileNotFoundError(
                f'Expected clean image not found: {clean_path}'
            )

        pairs.append({
            'degraded': os.path.join(degraded_dir, fname),
            'clean': clean_path,
            'deg_type': deg_type,
        })

    if not pairs:
        raise RuntimeError(f'No training pairs found under {data_dir}')

    return pairs


# ---------------------------------------------------------------------------
# Training dataset  (augmentation + random crop)
# ---------------------------------------------------------------------------

class TrainDataset(Dataset):
    """
    Paired degraded / clean dataset for training.

    Accepts the pre-built pair list from :func:`build_pairs` so the
    train / val split can be done externally before construction.
    """

    def __init__(self, pairs: list, patch_size: int = 128):
        """
        Args:
            pairs:      List of dicts from :func:`build_pairs`.
            patch_size: Square random-crop size fed to the model.
        """
        super().__init__()
        self.pairs = pairs
        self.patch_size = patch_size
        n_rain = sum(1 for p in pairs if p['deg_type'] == 'rain')
        n_snow = len(pairs) - n_rain
        print(f'[TrainDataset] {len(pairs)} pairs  (rain={n_rain}, snow={n_snow})')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        degraded_img = Image.open(pair['degraded']).convert('RGB')
        clean_img = Image.open(pair['clean']).convert('RGB')

        # Dimensions must be multiples of 16 for PromptIR
        degraded_img = crop_to_multiple(degraded_img)
        clean_img = crop_to_multiple(clean_img)

        # Synchronised random crop
        i, j, h, w = self._random_crop_params(degraded_img)
        degraded_img = TF.crop(degraded_img, i, j, h, w)
        clean_img = TF.crop(clean_img, i, j, h, w)

        # Random flip / rotation augmentation
        degraded_img, clean_img = random_augment(degraded_img, clean_img)

        return TF.to_tensor(degraded_img), TF.to_tensor(clean_img)

    def _random_crop_params(self, img):
        w, h = img.size
        if h < self.patch_size or w < self.patch_size:
            return 0, 0, h, w
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        return top, left, self.patch_size, self.patch_size


# ---------------------------------------------------------------------------
# Validation dataset  (full image, no augmentation, exposes deg_type)
# ---------------------------------------------------------------------------

class ValDataset(Dataset):
    """
    Paired degraded / clean dataset for validation.

    No augmentation or random crop — images are used at full resolution
    (cropped to the nearest multiple of 16).  The degradation type string
    (``'rain'`` or ``'snow'``) is returned as a third element so per-type
    PSNR can be computed in the training loop.
    """

    def __init__(self, pairs: list):
        """
        Args:
            pairs: List of dicts from :func:`build_pairs`.
        """
        super().__init__()
        self.pairs = pairs
        n_rain = sum(1 for p in pairs if p['deg_type'] == 'rain')
        n_snow = len(pairs) - n_rain
        print(f'[ValDataset]   {len(pairs)} pairs  (rain={n_rain}, snow={n_snow})')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        degraded_img = Image.open(pair['degraded']).convert('RGB')
        clean_img = Image.open(pair['clean']).convert('RGB')

        degraded_img = crop_to_multiple(degraded_img)
        clean_img = crop_to_multiple(clean_img)

        return (
            TF.to_tensor(degraded_img),
            TF.to_tensor(clean_img),
            pair['deg_type'],   # 'rain' or 'snow'
        )


# ---------------------------------------------------------------------------
# Test dataset  (no ground truth)
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    """
    Dataset for the test set (no ground truth).

    Returns ``(image_tensor, filename)`` where ``filename`` is e.g. ``'0.png'``.
    """

    def __init__(self, data_dir: str):
        super().__init__()
        degraded_dir = os.path.join(data_dir, 'test', 'degraded')
        self.image_paths = []
        self.filenames = []

        for fname in sorted(
            os.listdir(degraded_dir),
            key=lambda f: int(os.path.splitext(f)[0]),
        ):
            if fname.lower().endswith('.png'):
                self.image_paths.append(os.path.join(degraded_dir, fname))
                self.filenames.append(fname)

        if not self.image_paths:
            raise RuntimeError(f'No test images found under {degraded_dir}')

        print(f'[TestDataset]  {len(self.image_paths)} test images.')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = crop_to_multiple(img)
        return TF.to_tensor(img), self.filenames[idx]