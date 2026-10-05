"""
Main Training Script for Indus-Keeladi CNN Project
Trains the classifier on Indus script signs and prepares for Keeladi validation
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.image_normalization import ImageNormalizer
from src.models.indus_classifier_cnn import IndusClassifierCNN


def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ── Figure-65 sign corrections (verified against the source PDFs in docs/) ──
#
# "THE INDUS SCRIPT - Recognition as an Alphabet", Fig. 65 lists 40 serial signs.
# Several serials list multiple P-numbers joined by "or"; those are ONE sign drawn
# in different ways (allographs). The class folders had split them apart, which
# created 5 duplicate near-identical classes and forced the model to separate a
# single sign into several. Merging gives the source's 40 primary core signs.
ALLOGRAPH_GROUPS = {
    "sign_18_P181_P187": ("sign_18_P181", "sign_18_P187"),
    "sign_21_P200_P209": ("sign_21_P200", "sign_21_P209"),
    "sign_28_P272_P371": ("sign_28_P272", "sign_28_P371"),
    "sign_30_P282_P285_P287": ("sign_30_P282", "sign_30_P285", "sign_30_P287"),
    # NOTE: serial 17 ("156 or 165") was already merged as sign_17_P156_P165.
    # NOTE: sign_12/13_P130* and sign_35/36_P341* are DIFFERENT serials that merely
    #       share a P-number - they are intentionally NOT merged.
}
_MEMBER_TO_GROUP = {m: g for g, ms in ALLOGRAPH_GROUPS.items() for m in ms}

# The Keeladi annexure (keeladi_indus.pdf p.63) numbers its INDUS signs with
# MAHADEVAN (M-1977) numbers, not P-2010 numbers. Verified by glyph shape:
#   * annexure "225" == Fig.65 serial 24 (M=225, P=219)  -> not serial 25 (P=225)
#   * annexure "307" == Fig.65 serial 18 (M=304|307, P=181|187)
# So the annexure images must be trained as those core classes. Applied ONLY to
# images loaded from indus_matched/, never to primary_core_signs/.
ANNEXURE_REMAP = {
    "sign_25_P225_Cross": "sign_24_P219",
    "sign_41_P307": "sign_18_P181_P187",
}

# Image extensions treated as training samples.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Classes whose name starts with this prefix are "not an Indus sign"
# rejection classes.  src/evaluate.py (REJECTION_PREFIX) already excludes
# these from match claims, so a prediction of one means "reject".
REJECTION_PREFIX = "zz_"


def load_fig65_real_glyphs(fig65_dir, class_names):
    """
    Load the REAL Fig.65 glyphs digitised by src/digitise_fig65.py and map
    each onto the class it depicts.

    Why this matters: the folder-per-class corpus is one hand-drawn glyph plus
    augmented clones of it, so the classifier has never seen an independently
    drawn allograph.  These 39 glyphs are a genuinely separate rendering of the
    same signs, which is the only thing that lifts real accuracy.

    Mapping is by Fig.65 SERIAL number, not by P-number: ALLOGRAPH_GROUPS has
    merged several P-numbers into one class ("181|187" -> sign_18_P181_P187),
    so P-numbers are ambiguous, while serials stay 1:1.  Returns a list of
    (class_name, path) pairs; serials with no matching class are skipped.
    """
    import json
    import re
    from pathlib import Path

    fig65_dir = Path(fig65_dir)
    manifest_path = fig65_dir / "manifest.json"
    if not manifest_path.exists():
        return []

    by_serial = {}
    for c in class_names:
        m = re.match(r"sign_(\d+)_", c)
        if m:
            by_serial[int(m.group(1))] = c

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs, skipped = [], []
    for _key, meta in sorted(manifest.items()):
        serial = int(meta["serial"])
        cls = by_serial.get(serial)
        if cls is None:
            skipped.append(serial)
            continue
        p = fig65_dir / meta["file"]
        if p.exists():
            pairs.append((cls, p))
    if skipped:
        logging.getLogger(__name__).info(
            f"fig65_real: no class for serials {sorted(set(skipped))} (skipped)")
    return pairs


def generate_rejection_images(n_per_class=60, size=64, seed=1234):
    """
    Synthesise "not an Indus sign" training images.

    A closed-set softmax over real sign classes STRUCTURALLY CANNOT say "this
    is not an Indus sign" -- it is forced to pick the least-bad real sign.
    That is what produced the 67% false-match rate on general Keeladi
    graffiti.  Adding explicit rejection classes gives the network somewhere
    to put probability mass for blank/noise/line-art input.

    These are generated procedurally (blank, noise, pure strokes, blurred
    blobs) rather than taken from data/processed/val, because the val
    negatives are the EVALUATION CONTROL for src/audit_validity.openset_audit.
    Training on them would destroy that audit.  Using them as training data
    would make the open-set audit meaningless.

    Returns (images, names) with images as float32 (n, size, size, 1) in [0,1].
    """
    rng = np.random.default_rng(seed)
    import cv2

    def blank():
        lvl = rng.uniform(0.0, 0.12)
        return np.full((size, size), lvl, np.float32)

    def noise():
        base = rng.uniform(0.0, 0.15, size=(size, size)).astype(np.float32)
        # salt & pepper speckle
        amp = rng.uniform(0.3, 1.0)
        m = rng.random((size, size)) < rng.uniform(0.05, 0.35)
        base[m] = amp
        return np.clip(base, 0, 1)

    def strokes():
        """Random line art: not a real sign, but has stroke-like structure."""
        img = np.zeros((size, size), np.float32)
        for _ in range(rng.integers(2, 6)):
            p0 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            v = float(rng.uniform(0.6, 1.0))
            cv2.line(img, p0, p1, v, int(rng.integers(1, 4)))
        return img

    def blob():
        """Smooth out-of-focus smudge."""
        img = np.zeros((size, size), np.float32)
        for _ in range(rng.integers(1, 4)):
            c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            ax = int(rng.integers(size // 8, size // 2))
            cv2.ellipse(img, c, (ax, int(ax * rng.uniform(0.4, 1.0))),
                        float(rng.uniform(0, 180)), 0, 360,
                        float(rng.uniform(0.5, 1.0)), -1)
        k = int(rng.choice([5, 9, 15]))
        return cv2.GaussianBlur(img, (k, k), 0)

    kinds = {
        "zz_blank": blank,
        "zz_noise": noise,
        "zz_strokes": strokes,
        "zz_blob": blob,
    }
    imgs, names = [], []
    for nm, fn in kinds.items():
        for _ in range(n_per_class):
            a = np.clip(np.asarray(fn(), np.float32), 0.0, 1.0)
            imgs.append(a[..., None])
            names.append(nm)
    return np.stack(imgs), names


def canonical_class(folder_name):
    """Folder name -> canonical sign class (applies the allograph merge)."""
    return _MEMBER_TO_GROUP.get(folder_name, folder_name)


class IndusKeeladiTrainer:
    """
    Main training pipeline for Indus script classification
    """
    
    def __init__(self, data_dir, model_dir, augment_factor=40):
        """
        Initialize trainer
        
        Args:
            data_dir: Root directory for data
            model_dir: Directory to save trained models
            augment_factor: How many augmented copies to generate per original image
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.augment_factor = augment_factor
        
        self.normalizer = ImageNormalizer(target_size=(64, 64))
        self.classifier = None
        self.class_names = []
        self.data_augmentation = self._build_augmentation_pipeline()
    
    def _build_augmentation_pipeline(self):
        """
        Domain-adaptation augmentation pipeline.

        Goal: make synthetic-clean Indus glyphs look like messy Keeladi
        graffiti. The accuracy gap was caused by the CNN only seeing
        perfect matplotib shapes during training, then being fed noisy
        white-bordered screenshots of potsherds at test time.

        Harshness tuned so after 40x augs the model sees the full
        spectrum: pristine → slightly rotated → sheared → dim →
        speckled → slightly blurred → random-cropped (scaled).
        """
        return keras.Sequential([
            # 1. Bigger rotation + translation (graffiti scratched at any angle)
            keras.layers.RandomRotation(0.28),          # was 0.15 -> ±~100°
            keras.layers.RandomTranslation(0.18, 0.18),  # was 0.10
            keras.layers.RandomZoom(0.22, 0.22),         # was 0.10

            # 2. Shear / affine — potsherds are curved surfaces
            keras.layers.RandomFlip("horizontal"),       # script direction unknown

            # 3. Intensity shifts — scans/screenshots have varying brightness
            keras.layers.RandomContrast(0.45),           # was 0.20
            keras.layers.RandomBrightness(0.35, value_range=(0.0, 1.0)),

            # 4. Noise — photo compression, scan dust, JPG artifacts
            keras.layers.GaussianNoise(0.09),            # was 0.05

            # (Blur and salt-pepper are applied in NumPy below because
            #  Keras doesn't ship built-in Poisson/S&P layers.)
        ])

    def _numpy_stochastic_degrade(self, img: np.ndarray) -> np.ndarray:
        """
        Additional probability-based degradation applied per-augmented-image:
          - 30% chance of gaussian blur (potsherd out-of-focus)
          - 20% chance of salt & pepper noise (bad scan)
          - 25% chance of slight posterize (reduced dynamic range — web screenshot)
        Operates on H×W×C float32 image in [0, 1] range, returns same dtype.
        """
        import cv2
        x = img.copy()
        if x.ndim == 2:  # cv2 ops may drop the singleton channel dim
            x = x[..., np.newaxis]
        if np.random.rand() < 0.30:
            k = np.random.choice([3, 5])
            x = cv2.GaussianBlur(x, (k, k), sigmaX=0.5 + np.random.rand())
        if np.random.rand() < 0.20:
            # salt & pepper
            s_vs_p = 0.5
            amount = np.random.uniform(0.01, 0.08)
            salt = np.random.choice([0, 1], size=x.shape[:2],
                                    p=[1 - amount, amount]).astype(bool)
            pepper = np.random.choice([0, 1], size=x.shape[:2],
                                      p=[1 - amount * s_vs_p,
                                         amount * s_vs_p]).astype(bool)
            x[salt] = 1.0
            x[pepper] = 0.0
        if np.random.rand() < 0.25:
            # Slight posterize / quantize levels
            levels = np.random.choice([16, 32, 64])
            x = np.clip(np.round(x * levels) / levels, 0.0, 1.0)
        if np.random.rand() < 0.35:
            # Stroke-thickness shift - reference drawings are thick-stroked,
            # potsherd graffiti is thin-scratched, so erode/dilate the glyph
            k = int(np.random.choice([2, 3]))
            kern = np.ones((k, k), np.uint8)
            u8 = np.clip(x[..., 0] * 255.0, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.5:
                u8 = cv2.erode(u8, kern)
            else:
                u8 = cv2.dilate(u8, kern)
            x[..., 0] = u8.astype(np.float32) / 255.0
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    
    def _augment_dataset(self, X, y):
        """
        Augment the dataset by generating augmented copies of each image
        
        Args:
            X: Input images (N, H, W, C)
            y: Labels (N,)
            
        Returns:
            Augmented X and y arrays
        """
        logger = logging.getLogger(__name__)
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            # Keep original image
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            
            # Generate augmented copies
            img = np.expand_dims(X[i], axis=0)
            for j in range(self.augment_factor):
                aug_img = self.data_augmentation(img, training=True)[0].numpy()
                aug_img = self._numpy_stochastic_degrade(aug_img)
                # Keras 3 randomly drops the singleton channel dim;
                # restore it so all samples share the (64, 64, 1) shape
                aug_img = aug_img.reshape(64, 64, 1)
                augmented_X.append(aug_img)
                augmented_y.append(y[i])
        
        X_aug = np.array(augmented_X)
        y_aug = np.array(augmented_y)
        
        logger.info(f"Dataset augmented: {len(X)} -> {len(X_aug)} samples ({self.augment_factor}x factor)")
        return X_aug, y_aug
    
    def _safe_train_val_split(self, X, y, test_size=0.2):
        """
        Safely split data avoiding stratify errors for single-sample classes
        
        Args:
            X: Images array
            y: Labels array
            test_size: Fraction for validation set
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        logger = logging.getLogger(__name__)
        
        # Check if any class has only 1 sample
        unique_classes, counts = np.unique(y, return_counts=True)
        min_count = np.min(counts)
        classes_with_one = unique_classes[counts == 1]
        
        if min_count < 2:
            logger.warning(f"{len(classes_with_one)} classes have only 1 sample. Using non-stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )
        else:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y, shuffle=True
                )
            except Exception as e:
                logger.warning(f"Stratified split failed ({e}). Using non-stratified split.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42, shuffle=True
                )
        
        return X_train, X_val, y_train, y_val

    def _safe_index_split(self, y, test_size=0.2, random_state=42):
        """
        Return (train_idx, val_idx) for a stratified split that is safe for
        classes with only one sample. Used so we can split BEFORE augmenting.
        """
        idx = np.arange(len(y))
        _uniq, counts = np.unique(y, return_counts=True)
        if np.min(counts) < 2:
            return train_test_split(idx, test_size=test_size,
                                    random_state=random_state, shuffle=True)
        try:
            return train_test_split(idx, test_size=test_size,
                                    random_state=random_state, stratify=y)
        except Exception:
            return train_test_split(idx, test_size=test_size,
                                    random_state=random_state, shuffle=True)

    def load_training_data(self, augment=True, split_protocol="leakage_controlled"):
        """
        Load and preprocess training data from directory structure

        Args:
            augment: Whether to apply data augmentation
            split_protocol:
              "leakage_controlled" (default) - split the ORIGINAL images into
                  train/val FIRST, then augment ONLY the training split. This
                  removes the train/val clone overlap that used to inflate the
                  reported validation accuracy (see src/audit_validity.py).
              "stratified" - legacy behaviour (augment then split). Kept only
                  for reproducing the old, leaky numbers; emits a warning.

        Returns:
            X_train, y_train, X_val, y_val
        """
        logger = logging.getLogger(__name__)
        train_dir = self.data_dir / "processed" / "train" / "primary_core_signs"

        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        # Additional reference tree: user-curated Indus signs that have a known
        # Keeladi match (indus_matched/).  Subfolders are merged BY CLASS NAME
        # with primary_core_signs (e.g. sign_25_P225_Cross gains extra images).
        extra_dirs = [train_dir]
        matched_dir = self.data_dir / "processed" / "train" / "indus_matched"
        if matched_dir.exists():
            extra_dirs.append(matched_dir)

        # ------------------------------------------------------------------
        # Resolve folders -> canonical sign classes.
        #   * ALLOGRAPH_GROUPS merges Fig.65 "or" variants (45 -> 40 core signs):
        #     sign_18_P181/P187, sign_21_P200/P209, sign_28_P272/P371 and
        #     sign_30_P282/P285/P287 are ONE sign each, not four/five.
        #   * ANNEXURE_REMAP moves the Keeladi annexure images (which use
        #     MAHADEVAN numbering, verified against docs/*.pdf) into the core
        #     class they actually belong to, removing the label noise where two
        #     different glyphs shared the class sign_25_P225_Cross.
        # This is a load-time remap only: nothing on disk is changed.
        # ------------------------------------------------------------------
        canon_to_members = {}          # canonical class -> [(dir, folder_name)]
        for d in extra_dirs:
            is_matched = (d.name == "indus_matched")
            for sub in sorted(d.iterdir()):
                if not sub.is_dir():
                    continue
                cls = canonical_class(sub.name)
                if is_matched and sub.name in ANNEXURE_REMAP:
                    cls = ANNEXURE_REMAP[sub.name]
                # Skip folders that contain no images.  An empty folder (e.g.
                # sign_40_P120_SemiSigns) would otherwise register a class that
                # owns a softmax output but has no training signal, so it can
                # never be predicted and only dilutes the real classes.
                has_images = any(
                    p.suffix.lower() in _IMAGE_EXTS
                    for p in sub.iterdir() if p.is_file()
                )
                if not has_images:
                    logger.warning(f"Skipping empty class folder (no images): {sub}")
                    continue
                canon_to_members.setdefault(cls, []).append((d, sub.name))

        self.class_names = sorted(canon_to_members)
        n_real = len(self.class_names)
        logger.info(f"Found {n_real} classes "
                    f"(after Fig.65 allograph merge + annexure remap)")

        images = []
        labels = []
        groups = []          # source-group id per image (its canonical class)
        class_counts = []
        fig65_is_real = []   # indices of the real Fig.65 glyphs within images

        # Independently drawn REAL Fig.65 glyphs (src/digitise_fig65.py).
        # These add genuine drawing diversity: every folder-per-class image is
        # an augmented clone of ONE hand-drawn glyph, so without these the
        # classifier has never seen a second, independent rendering of a sign.
        fig65_pairs = load_fig65_real_glyphs(
            self.data_dir / "processed" / "train" / "fig65_real",
            self.class_names,
        )
        if fig65_pairs:
            fig65_by_class = {}
            for cls_name, p in fig65_pairs:
                fig65_by_class.setdefault(cls_name, []).append(p)
            logger.info(f"Adding {len(fig65_pairs)} real Fig.65 glyph(s) "
                        f"across {len(fig65_by_class)} classes "
                        f"(independent drawings, added to TRAIN only)")
        else:
            fig65_by_class = {}
            logger.info("fig65_real: no manifest found, skipping real glyphs")

        # Load images for each canonical class across all of its member folders
        for class_idx, class_name in enumerate(self.class_names):
            image_files = []
            for d, folder_name in canon_to_members[class_name]:
                class_dir = d / folder_name
                image_files += (
                    list(class_dir.glob("*.png")) +
                    list(class_dir.glob("*.jpg")) +
                    list(class_dir.glob("*.jpeg")) +
                    list(class_dir.glob("*.bmp"))
                )
            # Real Fig.65 glyphs for this class.  Flagged so the split below can
            # hold them out of validation: they are a single independent
            # drawing each, and validating on them would measure memorisation of
            # that one drawing rather than generalisation.
            fig65_files = fig65_by_class.get(class_name, [])
            n_images = len(image_files)
            class_counts.append((class_name, n_images))
            logger.info(f"  {class_name}: {n_images} image(s)"
                        + (f" + {len(fig65_files)} real Fig.65" if fig65_files else ""))
            for image_file in image_files:
                try:
                    processed_image = self.normalizer.process_image(image_file)
                    images.append(processed_image)
                    labels.append(class_idx)
                    groups.append(class_name)
                except Exception as e:
                    logger.error(f"Error loading {image_file}: {e}")
            for image_file in fig65_files:
                try:
                    processed_image = self.normalizer.process_image(image_file)
                    images.append(processed_image)
                    labels.append(class_idx)
                    groups.append(class_name)
                    fig65_is_real.append(len(images) - 1)
                except Exception as e:
                    logger.error(f"Error loading real glyph {image_file}: {e}")
        
        if len(images) == 0:
            raise RuntimeError("No training images were loaded. Check the dataset directory.")

        X = np.array(images)
        y = np.array(labels)

        # Reshape for CNN (add channel dimension)
        X = X.reshape(X.shape[0], 64, 64, 1)

        logger.info(f"Total raw training samples: {len(X)}")
        fig65_real_idx = np.asarray(sorted(fig65_is_real), dtype=np.int64)

        # ------------------------------------------------------------------
        # REJECTION ("not an Indus sign") CLASSES
        # Appended AFTER the real classes so every real class keeps its
        # original index -- evaluate.py resolves expected class names via the
        # saved *_classes.txt, and the audit's EXPECTED_MATCH_MAP depends on
        # those names being stable.
        # ------------------------------------------------------------------
        n_rej_per = int(os.environ.get("INDUS_REJECTION_PER_CLASS", "60"))
        if n_rej_per > 0:
            rej_X, rej_names = generate_rejection_images(n_per_class=n_rej_per)
            rej_classes = sorted(set(rej_names))
            base = len(self.class_names)
            remap = {nm: base + i for i, nm in enumerate(rej_classes)}
            self.class_names = self.class_names + rej_classes
            y_rej = np.asarray([remap[nm] for nm in rej_names], dtype=y.dtype)
            X = np.concatenate([X, rej_X.astype(X.dtype)], axis=0)
            y = np.concatenate([y, y_rej], axis=0)
            logger.info(
                f"Added {len(rej_classes)} rejection classes "
                f"({n_rej_per} images each): {rej_classes}")
            logger.info(f"Total classes now: {len(self.class_names)} "
                        f"({n_real} real + {len(rej_classes)} rejection)")

        # ------------------------------------------------------------------
        # LEAKAGE-CONTROLLED SPLIT
        # Old behaviour augmented FIRST and split SECOND, so augmented clones
        # of the same source glyph landed in both train and val; the reported
        # "validation accuracy" mostly measured memorisation (measured by
        # src.audit_validity.leakage_audit). We now split the ORIGINAL images
        # first and augment ONLY the training split.
        # ------------------------------------------------------------------
        if split_protocol == "stratified":
            logger.warning(
                "split_protocol='stratified' reproduces the OLD leaky behaviour "
                "(augment-then-split); reported accuracy will be inflated. "
                "Use 'leakage_controlled' for honest numbers.")
            if augment:
                X, y = self._augment_dataset(X, y)
            X_train, X_val, y_train, y_val = self._safe_train_val_split(X, y, test_size=0.15)
        else:
            tr_idx, va_idx = self._safe_index_split(y, test_size=0.15)
            tr_idx, va_idx = np.asarray(tr_idx), np.asarray(va_idx)
            # The real Fig.65 glyphs are single independent drawings. Keep them
            # in TRAIN and out of validation: with only one drawing per sign,
            # validating on it measures memorisation of that drawing, not
            # generalisation to a new allograph.
            if len(fig65_real_idx):
                va_set = set(va_idx.tolist())
                tr_set = set(tr_idx.tolist())
                moved = [i for i in fig65_real_idx.tolist()
                         if i in va_set and i not in tr_set]
                va_idx = np.asarray([i for i in va_idx.tolist() if i not in set(moved)],
                                    dtype=np.int64)
                tr_idx = np.concatenate([tr_idx, np.asarray(moved, dtype=np.int64)])
                tr_idx.sort()
                logger.info(
                    f"Real Fig.65 glyphs: moved {len(moved)} from val into train "
                    f"(held out of validation on purpose)")
            X_train, y_train = X[tr_idx], y[tr_idx]
            X_val, y_val = X[va_idx], y[va_idx]
            if augment:
                X_train, y_train = self._augment_dataset(X_train, y_train)
            logger.info(
                "Leakage-controlled split: originals split first; augmentation "
                "applied to the TRAIN split only (val images are never augmented).")

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")

        # Log class distribution summary
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        val_unique, val_counts = np.unique(y_val, return_counts=True)
        logger.info(f"Train classes: {len(train_unique)}, Val classes: {len(val_unique)}")

        return X_train, y_train, X_val, y_val
    
    def build_model(self, num_classes=None):
        """
        Build the classifier model
        
        Args:
            num_classes: Number of classes (uses len(class_names) if None)
        """
        logger = logging.getLogger(__name__)
        
        if num_classes is None:
            num_classes = len(self.class_names)
        
        self.classifier = IndusClassifierCNN(
            input_shape=(64, 64, 1),
            num_classes=num_classes
        )
        self.classifier.build_model()
        logger.info("Model built successfully")
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=80, batch_size=16):
        """
        Train the classifier
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training for {epochs} epochs (batch_size={batch_size})...")
        
        # Adjust epochs for very small datasets
        if len(X_train) < 200:
            epochs = max(epochs, 60)
            logger.info(f"Small dataset detected. Using {epochs} epochs for adequate learning.")
        
        history = self.classifier.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Log final metrics
        if hasattr(history, 'history'):
            h = history.history
            if 'accuracy' in h and 'val_accuracy' in h:
                final_acc = h['accuracy'][-1]
                final_val_acc = h['val_accuracy'][-1]
                best_val_acc = max(h['val_accuracy'])
                logger.info(f"Final training accuracy: {final_acc:.4f}")
                logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
                logger.info(f"Best validation accuracy: {best_val_acc:.4f} (epoch {np.argmax(h['val_accuracy'])+1})")
        
        return history
    
    def save_trained_model(self, model_name="indus_classifier"):
        """
        Save the trained model
        
        Args:
            model_name: Name for the saved model
        """
        logger = logging.getLogger(__name__)
        
        model_path = self.model_dir / f"{model_name}.keras"
        self.classifier.save_model(str(model_path))
        
        # Save class names
        class_names_path = self.model_dir / f"{model_name}_classes.txt"
        with open(class_names_path, 'w') as f:
            f.write('\n'.join(self.class_names))
        
        logger.info(f"Model and class names saved to {self.model_dir}")
    
    def load_trained_model(self, model_name="indus_classifier"):
        """
        Load a previously trained model
        
        Args:
            model_name: Name of the saved model
        """
        logger = logging.getLogger(__name__)
        
        model_path = self.model_dir / f"{model_name}.keras"
        class_names_path = self.model_dir / f"{model_name}_classes.txt"
        
        self.classifier.load_model(str(model_path))
        
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Loaded {len(self.class_names)} classes")


def main():
    """Main training pipeline"""
    
    # Set paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    model_dir = project_root / "models"
    log_dir = data_dir / "results" / "logs"
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("=" * 60)
    logger.info("INDUS-KEELADI CNN TRAINING PIPELINE STARTED")
    logger.info("=" * 60)
    
    # Schedule is configurable so a fast CPU run and a full-quality run can both
    # be reproduced without editing code:
    #   INDUS_AUG    augmented copies per image (default 25)
    #   INDUS_EPOCHS training epochs          (default 80)
    #   INDUS_BATCH  batch size               (default 16)
    aug_factor = int(os.environ.get("INDUS_AUG", "25"))
    n_epochs = int(os.environ.get("INDUS_EPOCHS", "80"))
    batch_size = int(os.environ.get("INDUS_BATCH", "16"))
    logger.info(f"Schedule: augment={aug_factor}x epochs={n_epochs} batch={batch_size}")

    # Initialize trainer
    trainer = IndusKeeladiTrainer(data_dir, model_dir, augment_factor=aug_factor)
    
    # Load training data with augmentation
    logger.info("Loading training data with augmentation...")
    try:
        X_train, y_train, X_val, y_val = trainer.load_training_data(augment=True)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise
    
    # Build model
    logger.info("Building CNN model...")
    trainer.build_model()
    
    # Display model summary
    trainer.classifier.get_model_summary()
    
    # Train model
    logger.info("Beginning model training...")
    history = trainer.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=n_epochs,
        batch_size=batch_size
    )
    
    # Save trained model
    logger.info("Saving trained model...")
    trainer.save_trained_model("indus_classifier")
    
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
