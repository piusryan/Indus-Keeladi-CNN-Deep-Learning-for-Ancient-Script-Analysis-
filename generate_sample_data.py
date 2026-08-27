"""
Generate realistic augmented variants for demonstration
Expands each 1-image class to ~20 images by applying controlled
geometric/photometric transforms to the REAL glyph, instead of
drawing random shapes. Keeps originals, adds aug_*.png files.
Safe to re-run: skips if class already has >=20 images.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import shutil

TARGET_PER_CLASS = 20
RNG = np.random.default_rng(42)

def augment_image(img: np.ndarray, seed: int) -> np.ndarray:
    """Apply a single realistic augmentation to a grayscale image."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    out = img.copy()

    # Random rotation ±18 degrees + scale 0.85-1.15 + translation ±4px
    angle = float(rng.uniform(-18, 18))
    scale = float(rng.uniform(0.85, 1.15))
    tx = float(rng.uniform(-4, 4))
    ty = float(rng.uniform(-4, 4))
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Brightness/contrast jitter
    alpha = float(rng.uniform(0.75, 1.25))  # contrast
    beta = float(rng.uniform(-18, 18))     # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # Stroke thickness: erode or dilate 30% chance
    if rng.random() < 0.30:
        k = int(rng.choice([2, 3]))
        kern = np.ones((k, k), np.uint8)
        if rng.random() < 0.5:
            out = cv2.erode(out, kern, iterations=1)
        else:
            out = cv2.dilate(out, kern, iterations=1)

    # Mild blur 25% chance
    if rng.random() < 0.25:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)

    # Salt & pepper 15% chance
    if rng.random() < 0.15:
        amt = float(rng.uniform(0.01, 0.04))
        mask = rng.random((h, w)) < amt
        out[mask] = 255 if rng.random() < 0.5 else 0

    # Horizontal flip 10% (script direction unknown)
    if rng.random() < 0.10:
        out = cv2.flip(out, 1)

    return out


def expand_folder(folder: Path, target: int = TARGET_PER_CLASS):
    """Expand one class folder to `target` images by augmenting existing files."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    existing = [p for p in folder.iterdir() if p.suffix.lower() in exts and not p.name.startswith("aug_")]
    if not existing:
        return 0
    current_total = len([p for p in folder.iterdir() if p.suffix.lower() in exts])
    if current_total >= target:
        print(f"  {folder.name}: {current_total} images (already >= {target}, skipping)")
        return 0

    needed = target - current_total
    # Use first real image as source (or round-robin if multiple)
    created = 0
    for i in range(needed):
        src = existing[i % len(existing)]
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Resize to 64x64 if needed, keep as is otherwise
        if img.shape[0] != 64 or img.shape[1] != 64:
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        aug = augment_image(img, seed=1000 + i + hash(folder.name) % 10000)
        out_path = folder / f"aug_{i:02d}.png"
        # Avoid overwrite if re-running
        if out_path.exists():
            out_path = folder / f"aug_{i:02d}_{random.randint(100,999)}.png"
        cv2.imwrite(str(out_path), aug)
        created += 1
    print(f"  {folder.name}: {current_total} -> {current_total+created} (+{created} augmented)")
    return created


def create_sample_dataset(target_per_class: int = TARGET_PER_CLASS):
    """Expand all training class folders to target_per_class images."""
    project_root = Path(__file__).parent
    train_roots = [
        project_root / "data" / "processed" / "train" / "primary_core_signs",
        project_root / "data" / "processed" / "train" / "indus_matched",
        # Also expand val match folders slightly so 1-image val isn't trivial
        # (only up to 5 so evaluation still tests generalization)
    ]
    total_created = 0
    for root in train_roots:
        if not root.exists():
            print(f"Missing: {root}")
            continue
        print(f"\nExpanding {root}:")
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                total_created += expand_folder(sub, target=target_per_class)

    # Optionally expand general graffiti a bit (keep small for testing invariance)
    val_general = project_root / "data" / "processed" / "val" / "keeladi" / "general_keeladi_graffiti"
    if val_general.exists():
        # Don't expand val too much; just ensure at least 4 remain
        pass

    print(f"\nDone. Total augmented images created: {total_created}")
    print(f"Each class now has ~{target_per_class} images (original + aug_*).")
    print("Re-run training to use expanded dataset.")


if __name__ == "__main__":
    create_sample_dataset()
