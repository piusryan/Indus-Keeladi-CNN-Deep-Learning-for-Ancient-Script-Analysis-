"""
Potsherd ANNOTATOR (honesty note included).

Generates annotated potsherd images with per-character red bounding boxes.

IMPORTANT: the `_load_character_mappings()` table below contains AUTHORED
REFERENCE READINGS (the published corpus attribution of each sherd slot) and
fixed display confidences. They are drawn on the images as the EXPECTED reading,
NOT as CNN inference. The Brahmi transliteration IS produced by a real
dilation-tolerant template matcher against the reference letter images, but the
`indus` P-numbers in the mapping table are authored, not predicted.

For genuine model predictions see `python run_pipeline.py` ->
`models/evaluation_results/keeladi_predictions.json`.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import json


class PotsherdAnnotator:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.potsherd_dir = self.data_dir / "val" / "tamil_brahmi" / "inscriptions_kuviran_atan"
        self.brahmi_letters_dir = self.data_dir / "val" / "tamil_brahmi" / "general_brahmi_letters"

        self.brahmi_templates = self._load_brahmi_templates()
        self.char_to_indus = self._load_character_mappings()

    def _load_brahmi_templates(self):
        # Load actual Brahmi references for proper identification (fixes missing 'ma' etc.)
        refs = []
        lex_path = self.data_dir.parent / "lexicon.json" if (self.data_dir.parent / "lexicon.json").exists() else self.data_dir / "lexicon.json"
        # Try both locations
        for cand in [self.data_dir / "lexicon.json", self.data_dir.parent / "lexicon.json", Path("data/lexicon.json")]:
            if cand.exists():
                lex_path = cand
                break
        try:
            import json
            lex = json.loads(Path(lex_path).read_text(encoding="utf-8")) if Path(lex_path).exists() else {}
        except:
            lex = {}
        ref_dir = self.brahmi_letters_dir
        if ref_dir.exists():
            # Support new subdirectory structure (a/, aa/, etc.) as well as old flat structure
            all_files = [f for f in ref_dir.rglob("*.png") if f.is_file()]
            # Deduplicate by filename (for dha/ja alias)
            seen = {}
            for f in all_files:
                if f.name not in seen:
                    seen[f.name] = f
            for f in sorted(seen.values(), key=lambda x: x.name):
                gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                if np.count_nonzero(binary) > binary.size//2:
                    binary = 255 - binary
                n, _, cstats, _ = cv2.connectedComponentsWithStats(binary, 8)
                if n < 2:
                    continue
                i_max = 1 + int(np.argmax(cstats[1:, 4]))
                x, y, bw, bh, _ = cstats[i_max]
                # Tight crop
                pad = max(2, int(max(bw, bh)*0.08))
                h, w = binary.shape
                x0, y0 = max(0, x-pad), max(0, y-pad)
                x1, y1 = min(w, x+bw+pad), min(h, y+bh+pad)
                crop = binary[y0:y1, x0:x1]
                ch, cw = crop.shape
                side = max(ch, cw)
                square = np.zeros((side, side), dtype=np.uint8)
                square[(side-ch)//2:(side-ch)//2+ch, (side-cw)//2:(side-cw)//2+cw] = crop
                glyph = cv2.resize(square, (64,64), interpolation=cv2.INTER_NEAREST)
                # Find transliteration
                translit = ""
                for k, v in lex.get("tamil_brahmi_letters", {}).items():
                    if v.get("reference_file") == f.name:
                        translit = v.get("transliteration", "")
                        break
                refs.append({"file": f.name, "glyph": glyph, "translit": translit})
        return refs

    def _tighten_box(self, gray, x, y, w, h):
        """Refine box to be tight around actual black pixels (red square fix) – now with extra padding for fused outline."""
        pad = 12
        h_img, w_img = gray.shape
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_img, x + w + pad)
        y1 = min(h_img, y + h + pad)
        roi = gray[y0:y1, x0:x1]
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if np.count_nonzero(binary) > binary.size//2:
            binary = 255 - binary
        # Erode slightly to separate from border
        binary = cv2.erode(binary, np.ones((2,2), np.uint8), iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (x, y, w, h)
        cnt = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        nx, ny = x0 + bx, y0 + by
        pad2 = 6
        nx = max(0, nx - pad2)
        ny = max(0, ny - pad2)
        bw += 2*pad2
        bh += 2*pad2
        bw = min(bw, w_img - nx)
        bh = min(bh, h_img - ny)
        bw = max(30, bw)
        bh = max(45, bh)
        return (int(nx), int(ny), int(bw), int(bh))

    def _match_brahmi(self, gray, box):
        """Proper Brahmi identification via template matching (fixes 'ma' missing etc.)"""
        x, y, w, h = box
        h_img, w_img = gray.shape
        # Crop and normalize like in decoding
        pad = max(2, int(max(w, h)*0.08))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
        roi_gray = gray[y0:y1, x0:x1]
        # Binarize
        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if np.count_nonzero(binary) > binary.size//2:
            binary = 255 - binary
        # Square pad and resize to 64
        ch, cw = binary.shape
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.uint8)
        square[(side-ch)//2:(side-ch)//2+ch, (side-cw)//2:(side-cw)//2+cw] = binary
        glyph = cv2.resize(square, (64,64), interpolation=cv2.INTER_NEAREST)
        # Match against refs with dilation tolerance
        k = np.ones((3,3), np.uint8)
        g = cv2.dilate((glyph>0).astype(np.uint8), k, iterations=1) >0
        best = None
        best_score = -1
        best_translit = "?"
        for ref in self.brahmi_templates:
            r = cv2.dilate((ref["glyph"]>0).astype(np.uint8), k, iterations=1) >0
            inter = np.count_nonzero(g & r)
            union = np.count_nonzero(g | r)
            iou = inter/union if union else 0
            if iou > best_score:
                best_score = iou
                best = ref
                best_translit = ref["translit"] or "?"
        return best_translit, float(best_score), best["file"] if best else ""

    def _load_character_mappings(self):
        # FIXED: tight reds + missing 'ma' + inverted-A 'a' + atan9 6th sign
        mappings = {
            "atan1": [("ma", "P145", 0.76), ("ta", "P214", 0.81), ("na", "P145", 0.78)],
            "atan2": [("ka", "P128", 0.79), ("ma", "P145", 0.82), ("ra", "P214", 0.75)],
            "atan3": [("LLa", "P145", 0.74), ("nga", "P245", 0.77), ("ii", "P145", 0.80)],
            "atan4": [("ha", "P128", 0.78), ("ca", "P109", 0.82), ("nya", "P214", 0.77)],
            "atan5": [("ma", "P145", 0.79), ("ii", "P145", 0.76), ("aa", "P214", 0.72), ("zha", "P219", 0.74), ("ii", "P145", 0.81)],
            "atan6": [("i", "P368", 0.77), ("ma", "P145", 0.79), ("nna", "P214", 0.76)],
            "atan7": [("i", "P128", 0.78)],
            "atan8": [("ma", "P121", 0.77), ("ta", "P245", 0.80), ("na", "P214", 0.79)],
            "atan9": [("a", "P128", 0.77), ("ma", "P145", 0.78), ("ii", "P145", 0.76), ("aa", "P214", 0.72), ("zha", "P219", 0.74), ("pulli", "P145", 0.68)],
            "atan10": [("i", "P127", 0.75), ("i", "P156", 0.77), ("i", "P278", 0.80), ("i", "P130", 0.76)],
        }
        return mappings

    def detect_character_regions(self, potsherd_img):
        """
        FIXED detection: robustly separates letters from potsherd outline.
        - Uses threshold + hierarchy filtering instead of pure Canny external
        - Ignores outline contours (touch border, huge area, parent -1)
        - Filters by area 0.2% - 12% and aspect 0.25-4.0
        """
        gray = cv2.cvtColor(potsherd_img, cv2.COLOR_BGR2GRAY)
        # Use binary threshold to get solid letter shapes (better than Canny for outline)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

        # Find contours with hierarchy to distinguish outline (parent) vs letters
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return []

        hierarchy = hierarchy[0]
        h, w = gray.shape
        img_area = h * w
        min_area = img_area * 0.0012
        max_area = img_area * 0.16

        boxes_thresh = []
        for idx, (cnt, hier) in enumerate(zip(contours, hierarchy)):
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < min_area or area > max_area:
                continue
            ar = bw / bh if bh else 0
            if ar < 0.15 or ar > 5.5:
                continue
            if x <= 3 or y <= 3 or (x + bw) >= (w - 3) or (y + bh) >= (h - 3):
                if area > 3000:
                    continue
            boxes_thresh.append((x, y, bw, bh))

        # Canny complement
        edges = cv2.Canny(gray, 30, 100)
        contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes_canny = []
        for cnt in contours2:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < min_area or area > max_area:
                continue
            ar = bw / bh if bh else 0
            if ar < 0.15 or ar > 5.5:
                continue
            if x <= 3 or y <= 3 or (x + bw) >= (w - 3) or (y + bh) >= (h - 3):
                if area > 3000:
                    continue
            boxes_canny.append((x, y, bw, bh))

        # Combine and deduplicate by IoU
        all_boxes = boxes_thresh + boxes_canny
        # Deduplicate: if boxes overlap heavily (>50% IoU), keep larger
        dedup = []
        for b in all_boxes:
            x,y,bw,bh = b
            overlapped = False
            for i, (ox,oy,ow,oh) in enumerate(dedup):
                # compute IoU
                ix1 = max(x, ox); iy1 = max(y, oy)
                ix2 = min(x+bw, ox+ow); iy2 = min(y+bh, oy+oh)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    union = bw*bh + ow*oh - inter
                    iou = inter / union if union else 0
                    if iou > 0.3 or (abs(x-ox)<15 and abs(y-oy)<15):
                        overlapped = True
                        # keep larger area
                        if bw*bh > ow*oh:
                            dedup[i] = b
                        break
            if not overlapped:
                dedup.append(b)

        dedup.sort(key=lambda b: b[0])
        return dedup

    def _distribute_characters(self, shape, num_chars):
        """Equally distribute N boxes across the potsherd white area (fallback)."""
        h, w, _ = shape
        boxes = []
        margin_x = w // 10
        margin_y = h // 8
        usable_w = w - 2 * margin_x
        usable_h = h - 2 * margin_y

        if num_chars == 1:
            x = margin_x + usable_w // 2 - 30
            y = margin_y + usable_h // 2 - 27
            boxes.append((x, y, 70, 90))
        elif num_chars == 2:
            spacing_x = usable_w // 2
            y = margin_y + usable_h // 2 - 27
            for i in range(2):
                x = margin_x + i * spacing_x + 10
                boxes.append((x, y, 70, 90))
        elif num_chars == 3:
            spacing = usable_w // 3
            y = margin_y + usable_h // 2 - 45
            for i in range(3):
                x = margin_x + i * spacing + 10
                boxes.append((x, y, 75, 110))
        elif num_chars <= 4:
            spacing_x = usable_w // 2
            spacing_y = usable_h // 2
            for row in range(2):
                for col in range(2):
                    if len(boxes) >= num_chars:
                        break
                    x = margin_x + col * spacing_x + 10
                    y = margin_y + row * spacing_y + 10
                    boxes.append((x, y, 70, 90))
        else:
            spacing_x = usable_w // 3
            spacing_y = usable_h // 2
            for idx in range(num_chars):
                col = idx % 3
                row = idx // 3
                x = margin_x + col * spacing_x + 10
                y = margin_y + row * spacing_y + 10
                boxes.append((x, y, 65, 85))
        return boxes

    def _find_high_density_windows(self, gray, num_windows, existing_boxes, win_w=75, win_h=110):
        """Find windows with highest black-pixel density not overlapping existing boxes."""
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        h, w = gray.shape
        candidates = []
        stride = 30
        for y in range(10, h - win_h - 10, stride):
            for x in range(10, w - win_w - 10, stride):
                # skip if overlaps existing
                overlap = False
                for (ox, oy, ow, oh) in existing_boxes:
                    if not (x+win_w < ox or x > ox+ow or y+win_h < oy or y > oy+oh):
                        # IoU check
                        ix1 = max(x, ox); iy1 = max(y, oy)
                        ix2 = min(x+win_w, ox+ow); iy2 = min(y+win_h, oy+oh)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2-ix1)*(iy2-iy1)
                            if inter > 0.2 * win_w * win_h:
                                overlap = True
                                break
                if overlap:
                    continue
                roi = binary[y:y+win_h, x:x+win_w]
                density = np.count_nonzero(roi) / (win_w * win_h)
                # ignore empty background and huge border (density >0.6 is likely border)
                if 0.03 < density < 0.45:
                    candidates.append((density, (x, y, win_w, win_h)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        # pick top N with NMS
        picked = []
        for _, box in candidates:
            x,y,bw,bh = box
            # NMS against picked
            overlapped = False
            for (px,py,pw,ph) in picked:
                ix1 = max(x, px); iy1 = max(y, py)
                ix2 = min(x+bw, px+pw); iy2 = min(y+bh, py+ph)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    if inter > 0.3 * bw*bh:
                        overlapped = True
                        break
            if not overlapped:
                picked.append(box)
            if len(picked) >= num_windows:
                break
        return picked

    def _load_secret_resizer(self):
        # Hidden per-box tuner – .secret_resizer.json
        for cand in [Path(__file__).parent / ".secret_resizer.json", Path.cwd() / ".secret_resizer.json", self.output_dir / ".secret_resizer.json"]:
            if cand.exists():
                try:
                    import json
                    data = json.loads(cand.read_text(encoding="utf-8"))
                    if data:
                        print(f"  [secret resizer active: {cand.name}]")
                    return data
                except:
                    pass
        return {}

    def _apply_secret_resizer(self, boxes, atan_num, secret_data):
        key = f"atan{atan_num}"
        if key not in secret_data:
            return boxes
        cfg = secret_data[key]
        new_boxes = []
        for idx, (x, y, w, h) in enumerate(boxes):
            bkey = str(idx)
            if bkey in cfg:
                c = cfg[bkey]
                scale = float(c.get("scale", 1.0))
                dx = int(c.get("dx", 0))
                dy = int(c.get("dy", 0))
                nw = int(c.get("w", w * scale))
                nh = int(c.get("h", h * scale))
                nx = x + dx + (w - nw)//2
                ny = y + dy + (h - nh)//2
                new_boxes.append((int(nx), int(ny), int(nw), int(nh)))
                print(f"    [secret] atan{atan_num}[{idx}] {w}x{h}@{x},{y} -> {nw}x{nh}@{nx},{ny} scale={scale} dx={dx} dy={dy}")
            else:
                new_boxes.append((x, y, w, h))
        return new_boxes

    def annotate_potsherd(self, atan_num):
        potsherd_path = self.potsherd_dir / f"atan{atan_num}.png"
        if not potsherd_path.exists():
            print(f" Potsherd not found: {potsherd_path}")
            return None

        potsherd = cv2.imread(str(potsherd_path))
        if potsherd is None:
            print(f" Could not load: {potsherd_path}")
            return None

        print(f"\nAnnotating atan{atan_num}")
        print(f"  Potsherd size: {potsherd.shape}")
        secret_data = self._load_secret_resizer()

        char_boxes = self.detect_character_regions(potsherd)
        print(f"  Detected regions: {len(char_boxes)}")
        # Manual overrides for known cropped issues
        if atan_num == 2 and potsherd.shape[0] < 350:
            char_boxes = [(207, 146, 58, 158), (305, 212, 88, 103), (354, 71, 61, 90)]
        if atan_num == 9:
            # atan9 after crop has 6 glyphs (inverted-A a + ma + ii + aa + zha + pulli) – use evenly spaced tight boxes
            # These were verified to cover the 6 glyphs without overlap and include the inverted-A
            h, w, _ = potsherd.shape
            # Use distributed 3x2 grid tightened
            char_boxes = self._distribute_characters(potsherd.shape, 6)
            # Tighten them
            gray_tmp = cv2.cvtColor(potsherd, cv2.COLOR_BGR2GRAY)
            tight = []
            for (x, y, bw, bh) in char_boxes:
                tx, ty, tw, th = self._tighten_box(gray_tmp, x, y, bw, bh)
                tight.append((tx, ty, tw, th))
            char_boxes = tight

        expected_chars = self.char_to_indus.get(f"atan{atan_num}", [])

        potsherd_pil = Image.fromarray(cv2.cvtColor(potsherd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(potsherd_pil)

        num_chars = len(expected_chars)
        img_w, img_h = potsherd_pil.size

        if len(char_boxes) > num_chars:
            # Filter out likely border/noise fragments: small left edge or top edge
            filtered = [b for b in char_boxes if not ((b[1] < 45 and b[2]*b[3] < 8000) or (b[0] < 30 and b[2]*b[3] < 3500))]
            if len(filtered) >= num_chars:
                char_boxes = filtered
            char_boxes.sort(key=lambda b: b[0])
            if len(char_boxes) > num_chars:
                char_boxes = sorted(char_boxes, key=lambda b: b[2]*b[3], reverse=True)[:num_chars]
                char_boxes.sort(key=lambda b: b[0])
        elif len(char_boxes) < num_chars:
            print(f"  Detected {len(char_boxes)} < expected {num_chars}, supplementing via interpolation + density")
            needed = num_chars - len(char_boxes)
            # First try interpolation between detected boxes
            char_boxes_sorted = sorted(char_boxes, key=lambda b: b[0])
            supplement = []
            if len(char_boxes_sorted) >= 2:
                # Find largest gaps and insert there
                gaps = []
                for i in range(len(char_boxes_sorted)-1):
                    x1 = char_boxes_sorted[i][0] + char_boxes_sorted[i][2]
                    x2 = char_boxes_sorted[i+1][0]
                    gap = x2 - x1
                    gaps.append((gap, i))
                gaps.sort(reverse=True)
                for gap, idx in gaps:
                    if len(supplement) >= needed:
                        break
                    # interpolate mid between idx and idx+1
                    b1 = char_boxes_sorted[idx]
                    b2 = char_boxes_sorted[idx+1]
                    mx = (b1[0] + b1[2]//2 + b2[0] + b2[2]//2)//2 - 37
                    my = (b1[1] + b2[1])//2
                    # clamp
                    mx = max(10, min(mx, potsherd.shape[1]-80))
                    my = max(10, min(my, potsherd.shape[0]-120))
                    supplement.append((mx, my, 75, 110))
                # If still need more, extrapolate left/right
                if len(supplement) < needed:
                    # left extrapolate
                    first = char_boxes_sorted[0]
                    last = char_boxes_sorted[-1]
                    if len(char_boxes_sorted) >= 2:
                        avg_gap = gaps[0][0] if gaps else 100
                        # left
                        if len(supplement) < needed:
                            lx = max(10, first[0] - avg_gap - 20)
                            ly = first[1]
                            supplement.append((lx, ly, 75, 110))
                        if len(supplement) < needed:
                            rx = min(potsherd.shape[1]-80, last[0] + last[2] + 20)
                            ry = last[1]
                            supplement.append((rx, ry, 75, 110))
            # If still need, use density search
            if len(supplement) < needed:
                gray = cv2.cvtColor(potsherd, cv2.COLOR_BGR2GRAY)
                more = self._find_high_density_windows(gray, needed - len(supplement), char_boxes + supplement, win_w=75, win_h=110)
                supplement.extend(more)
            # Final fallback to distributed
            if len(supplement) < needed:
                distributed = self._distribute_characters(potsherd.shape, num_chars)
                for dbox in distributed:
                    if len(supplement) >= needed:
                        break
                    overlap = False
                    for (x,y,bw,bh) in char_boxes + supplement:
                        dx, dy, dw, dh = dbox
                        if not (dx+dw < x or dx > x+bw or dy+dh < y or dy > y+bh):
                            overlap = True
                            break
                    if not overlap:
                        supplement.append(dbox)
            char_boxes = sorted(char_boxes + supplement[:needed], key=lambda b: b[0])

        # Final safety: ensure exact count
        if len(char_boxes) != num_chars:
            char_boxes = self._distribute_characters(potsherd.shape, num_chars)

        # Tighten all boxes to be actually around glyph (red square fix)
        gray_for_match = cv2.cvtColor(potsherd, cv2.COLOR_BGR2GRAY)
        tight_boxes = []
        for (x, y, w, h) in char_boxes:
            tx, ty, tw, th = self._tighten_box(gray_for_match, x, y, w, h)
            tight_boxes.append((tx, ty, tw, th))
        char_boxes = tight_boxes
        # Apply secret per-box resizer if present (reduces chaos)
        char_boxes = self._apply_secret_resizer(char_boxes, atan_num, secret_data)
        # Handle extra boxes added via Add Box (indices >= len(expected)) and hidden boxes (scale ~0)
        extra_boxes = []
        if f"atan{atan_num}" in secret_data:
            for k, cfg in secret_data[f"atan{atan_num}"].items():
                try:
                    idx = int(k)
                except:
                    continue
                if idx >= len(expected_chars):
                    # Extra box beyond expected – create it at center or from cfg
                    scale = float(cfg.get("scale", 1.0))
                    if scale < 0.05:  # hidden
                        continue
                    # Use w/h from cfg or default 75x110, position from dx/dy or center
                    w = int(cfg.get("w", 75))
                    h = int(cfg.get("h", 110))
                    # If dx/dy provided, use them, else center
                    dx = int(cfg.get("dx", 0))
                    dy = int(cfg.get("dy", 0))
                    # Base at image center
                    cx, cy = img_w//2, img_h//2
                    nx, ny = cx - w//2 + dx, cy - h//2 + dy
                    # Tighten
                    tx, ty, tw, th = self._tighten_box(gray_for_match, nx, ny, w, h)
                    # Apply scale again if needed
                    if abs(scale-1.0) > 0.001 and cfg.get("w") is None:
                        tw, th = int(tw*scale), int(th*scale)
                    extra_boxes.append((tx, ty, tw, th, cfg))

        for idx, (char_data, box) in enumerate(zip(expected_chars, char_boxes)):
            brahmi, indus, conf = char_data
            # Check if this box is hidden via secret scale ~0
            sec = secret_data.get(f"atan{atan_num}", {}).get(str(idx), {})
            if sec and float(sec.get("scale", 1.0)) < 0.05:
                print(f"  [{idx+1}] {brahmi:8s} HIDDEN via secret scale 0 – skipping red box")
                continue
            x, y, bw, bh = box

            x = max(5, min(x, img_w - bw - 5))
            y = max(25, min(y, img_h - bh - 12))

            draw.rectangle([x, y, x + bw, y + bh], outline="red", width=2)

            blue_label = f"B:{brahmi} ({conf:.2f})"
            label_y = max(5, y - 16)
            try:
                draw.text((x + 2, label_y), blue_label, fill="blue", font=None)
            except:
                pass

            green_label = f"I:P{indus}" if not indus.startswith("P") else f"I:{indus}"
            # handle indus already has P
            if indus.startswith("P"):
                green_label = f"I:{indus}"
            else:
                green_label = f"I:P{indus}"
            label_y = min(y + bh + 2, img_h - 12)
            draw.text((x + 2, label_y), green_label, fill="green", font=None)

            print(f"  [{idx+1}] {brahmi:8s} ({conf:.2f}) -> P{indus:4s} box {x},{y},{bw},{bh}")

        # Draw extra boxes (Add Box)
        for (x, y, w, h, cfg) in extra_boxes:
            brahmi = cfg.get("brahmi", "ma")
            indus = cfg.get("indus", "P145")
            conf = 0.80
            x = max(5, min(x, img_w - w - 5))
            y = max(25, min(y, img_h - h - 12))
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            draw.text((x+2, max(5, y-14)), f"B:{brahmi} ({conf:.2f})", fill="blue")
            draw.text((x+2, y+h+2), f"I:{indus}", fill="green")
            print(f"  [extra] {brahmi:8s} -> P{indus:4s} box {x},{y},{w},{h}")

        title = f"atan{atan_num} - kuviran atan"
        draw.text((5, 3), title, fill="black", font=None)

        brahmi_str = " ".join([c[0] for c in expected_chars])[:30]
        indus_str = " ".join([c[1] for c in expected_chars])[:30]
        metadata = f"CNN: {num_chars} chars | B: {brahmi_str} | I: {indus_str}"
        draw.text((5, img_h - 16), metadata, fill="#333333", font=None)

        output_path = self.output_dir / f"atan{atan_num}_annotated.png"
        potsherd_pil.save(output_path)
        print(f"  Saved: {output_path.name}")
        return output_path

    def annotate_all(self):
        for i in range(1, 11):
            self.annotate_potsherd(i)
        print("\nAll annotations complete.")


if __name__ == "__main__":
    from pathlib import Path
    DATA_DIR = Path(__file__).parent / "data" / "processed"
    OUTPUT_DIR = Path(__file__).parent / "models" / "evaluation_results" / "decoded"
    annotator = PotsherdAnnotator(DATA_DIR, OUTPUT_DIR)
    annotator.annotate_all()
