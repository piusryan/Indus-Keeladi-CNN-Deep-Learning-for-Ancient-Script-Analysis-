"""
Image Normalization Module for Indus-Keeladi CNN Project
Handles grayscale conversion, RESIZE + GLYPH AUTOCROP, DESKEW, denoise,
and standardization of sign images — applies SAME pipeline to TRAIN and VAL.

This is the accuracy-improved version: previously training data was clean
synthetic glyphs and val data was messy bordered screenshots with whitespace
around the graffiti. Now we autocrop to the glyph contour + deskew so the
CNN sees the actual shape filling the 64x64 canvas regardless of source.
"""

import cv2
import numpy as np
from pathlib import Path


class ImageNormalizer:
    """Normalizes Indus script and Keeladi graffiti images for CNN training"""

    def __init__(self, target_size=(64, 64), autocrop=True, deskew=True,
                 morph_clean=True):
        """
        Initialize normalizer with target image size and accuracy-improving
        preprocessing flags.

        Args:
            target_size: Tuple of (height, width) for output images
            autocrop: Find the glyph bounding-box and crop to it first
            deskew: Use image moments to correct rotational skew ±45°
            morph_clean: Apply morphological open/close for speckle noise
        """
        self.target_size = target_size
        self.autocrop = autocrop
        self.deskew = deskew
        self.morph_clean = morph_clean

    # ── I/O helpers ────────────────────────────────────────────────────
    def load_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image

    def convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def resize_image(self, image):
        return cv2.resize(image, (self.target_size[1], self.target_size[0]),
                          interpolation=cv2.INTER_AREA)

    def normalize_pixel_values(self, image):
        return image.astype(np.float32) / 255.0

    # ── New accuracy-improving preprocessing steps ─────────────────────
    def invert_if_dark_on_bright(self, gray):
        """
        Ensure glyph is DARK (low values) on BRIGHT background (high values)?
        Actually, we want the GLYPH to have HIGH pixel values (white-ish) on
        LOW background so the contour detection always picks the shape.
        We normalize towards 0 = background, 1 = glyph (convention used by
        training synthetic images).
        """
        # Take corners vs center; if center is darker than average corners,
        # glyph is dark-on-light, invert it.
        h, w = gray.shape
        corners = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
        avg_corner = float(np.mean(corners))
        center = float(gray[h // 2, w // 2])
        if center < avg_corner - 15:
            # Dark glyph on light bg -> invert
            return 255 - gray
        return gray

    def _denoise(self, gray):
        """Morphological open+close for salt/pepper or screenshot artifacts"""
        if not self.morph_clean:
            return gray
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k1)
        # Use median-blur as final mild denoise
        return cv2.medianBlur(closed, 3) if gray.shape[0] > 10 else closed

    def _autocrop_to_glyph(self, gray):
        """
        Detect the largest non-background contour, crop image to its
        bounding box, then add padding to return a square-ish crop.
        """
        if not self.autocrop:
            return gray
        h, w = gray.shape

        # Threshold (Otsu adapts to different lighting/screenshot darkness)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure glyph pixels are white (foreground) — pick by whichever side
        # has fewer white pixels after threshold
        if np.count_nonzero(binary) > (h * w) // 2:
            binary = 255 - binary

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return gray

        # Pick the contour with largest area
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 25:  # too small, probably noise, keep original
            return gray

        x, y, bw, bh = cv2.boundingRect(c)
        # Expand bbox slightly to avoid clipping thin strokes
        pad = max(2, int(max(bw, bh) * 0.08))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        cropped = gray[y0:y1, x0:x1]

        if cropped.shape[0] < 4 or cropped.shape[1] < 4:
            return gray

        # Pad cropped into a SQUARE so we don't distort shape on later resize
        ch, cw = cropped.shape
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.uint8)
        yoff = (side - ch) // 2
        xoff = (side - cw) // 2
        square[yoff:yoff + ch, xoff:xoff + cw] = cropped
        return square

    def _deskew(self, gray):
        """
        Use image moments to compute and correct ±45° of rotational skew.
        Good for graffiti scratched at slight angles on the potsherd.
        """
        if not self.deskew:
            return gray

        m = cv2.moments(gray)
        if abs(m['mu02']) < 1e-3:
            return gray

        # Skew angle via moments
        skew = m['mu11'] / m['mu02']
        if abs(skew) > 1.3:  # insane value, bad moments
            return gray

        h, w = gray.shape
        M = np.float32([[1, skew, -0.5 * w * skew],
                        [0, 1,    0]])
        flags = cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC
        return cv2.warpAffine(gray, M, (w, h), flags=flags,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

    def apply_threshold(self, image, threshold=127):
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary

    # ── Public combined pipelines ──────────────────────────────────────
    def process_image(self, image_path, apply_threshold=False):
        """
        Complete processing pipeline for a single image.

        **Order matters for accuracy**:
          load → grayscale → brightness-normalize invert → denoise →
          AUTOCROP (bbox + square pad) → DESKEW → resize → (optional thresh)
          → normalize [0..1]

        Returns:
            Processed and normalized image array (target_size H×W, float32 0..1)
        """
        image = self.load_image(image_path)
        gray = self.convert_to_grayscale(image)

        # Ensure consistent polarity: glyph bright, bg dark
        gray = self.invert_if_dark_on_bright(gray)
        # Remove screenshot / scan speckle
        gray = self._denoise(gray)
        # Crop to glyph contour (biggest accuracy improvement)
        gray = self._autocrop_to_glyph(gray)
        # Correct slight shear/rotation
        gray = self._deskew(gray)

        if apply_threshold:
            gray = self.apply_threshold(gray)

        gray = self.resize_image(gray)
        return self.normalize_pixel_values(gray)

    def process_array(self, img_array_bgr_or_gray, apply_threshold=False):
        """Same pipeline but takes an in-memory array (for live tests)."""
        gray = self.convert_to_grayscale(img_array_bgr_or_gray)
        gray = self.invert_if_dark_on_bright(gray)
        gray = self._denoise(gray)
        gray = self._autocrop_to_glyph(gray)
        gray = self._deskew(gray)
        if apply_threshold:
            gray = self.apply_threshold(gray)
        gray = self.resize_image(gray)
        return self.normalize_pixel_values(gray)

    def process_directory(self, input_dir, output_dir, apply_threshold=False):
        """
        Process all images in a directory (same consistent pipeline).
        Returns count of successfully processed images.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processed_count = 0
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

        for image_file in input_path.iterdir():
            if image_file.suffix.lower() not in image_extensions:
                continue
            try:
                processed = self.process_image(image_file, apply_threshold)
                output_file = output_path / image_file.name
                cv2.imwrite(str(output_file),
                            (processed * 255).astype(np.uint8))
                processed_count += 1
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        return processed_count


if __name__ == "__main__":
    # Example usage
    normalizer = ImageNormalizer(target_size=(64, 64))
    print("Image Normalizer initialized (autocrop+deskew+denoise enabled)")
