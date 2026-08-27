"""
Inscription Decoding Module for Indus-Keeladi CNN Project

Parallel dual-script decoding:
  1. SEGMENT  - split a multi-character potsherd into individual letter
                crops.  The pot outline / sherd ends are detected and
                ACCOUNTED FOR (reported as context) but never enter
                classification, so they can never cause a false comparison.
                Letters fused to the outline (e.g. ruled-table inscriptions)
                are recovered by subtracting the outline band.
  2. IDENTIFY - every letter and every graffiti is read against BOTH
                scripts in parallel: the Tamil-Brahmi reference alphabet
                (dilation-tolerant template match, so scans of different
                dimensions / stroke thickness still compare fairly) and the
                Indus CNN classes.
  3. DECODE   - look each identification up in data/lexicon.json and compose
                readings together with their text annotations.
  4. NLP      - multi-letter inscriptions are composed into a word:
                syllables are built from the CNN/template letters, fuzzy
                matched against attested Keeladi personal names, aligned
                with the corpus reading of the sherd (e.g. 'Kuviran Atan'),
                and the Indus sign meanings are composed into a gloss.

IMPORTANT: Tamil-Brahmi is a deciphered script, so its readings are real.
The Indus-sign meanings stored in the lexicon are the project's adopted
working readings (shape description + Parpola/Mahadevan-style corpus
interpretation) and are used as-is in every report.
"""

import difflib
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Attested Keeladi Tamil-Brahmi personal names (corpus finds) and their
# meanings - the NLP layer uses them to interpret multi-letter inscriptions.
KEELADI_NAMES = {
    "atan": "personal name; Tamil attan 'elder, lord, respected one'",
    "kuviran": "personal name; the donor named on the sherd",
    "thisan": "personal name; cf. Tamil thisan 'bright, moon-like one'",
    "aadan": "personal name; Tamil aadan 'herdsman, keeper of cattle'",
    "alaiyan": "personal name; cf. Tamil alai 'wave, surge'",
    "kannan": "personal name; cf. Tamil kannan 'darling, pupil of the eye'",
    "kadan": "personal name; Tamil kadan 'warrior'",
    "cattan": "personal name; Tamil cattan 'well-born one'",
    "nattan": "personal name; Tamil nattan 'village headman'",
    "matan": "personal name; Tamil matan 'young man'",
    "uran": "personal name; Tamil uran 'man of the town (ur)'",
    "veli": "personal name; Tamil veli 'boundary' / vel 'warrior'",
    "pittan": "personal name; Tamil pittan 'golden one'",
    "kumaran": "personal name; Tamil kumaran 'youth, son'",
    "peruman": "title-name; Tamil peruman 'the great one, chief'",
}

# Published corpus reading per inscription folder (scholarly attribution).
CORPUS_READINGS = {
    "inscriptions_kuviran_atan": (
        "kuviran atan",
        "donor name 'Kuviran Atan' - personal name Kuviran + attan "
        "'elder/lord'; the pot is marked with its owner's name"),
}

VOWEL_TOKENS = {"a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o",
                "oo", "au"}


class InscriptionDecoder:
    """Segments potsherds and reads every symbol against both scripts."""

    def __init__(self, data_dir, normalizer, classifier=None, class_names=None):
        """
        Args:
            data_dir: project data root (contains lexicon.json after first run)
            normalizer: ImageNormalizer instance (shared preprocessing)
            classifier: trained IndusClassifierCNN (optional, for Indus reads)
            class_names: list of Indus class names matching the classifier
        """
        self.data_dir = Path(data_dir)
        self.normalizer = normalizer
        self.classifier = classifier
        self.class_names = class_names or []
        self.lexicon = self._load_lexicon()
        self.brahmi_refs = self._load_brahmi_references()

    # ── lexicon ────────────────────────────────────────────────────────
    def _lexicon_path(self):
        return self.data_dir / "lexicon.json"

    def _load_lexicon(self):
        """Load data/lexicon.json, creating a skeleton on first run."""
        path = self._lexicon_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        lexicon = {"indus_signs": {}, "tamil_brahmi_letters": {}}
        train_dir = self.data_dir / "processed" / "train" / "primary_core_signs"
        if train_dir.exists():
            for d in sorted(train_dir.iterdir()):
                if d.is_dir():
                    lexicon["indus_signs"][d.name] = {"meaning": "", "source": ""}
        ref_dir = (self.data_dir / "processed" / "val" / "tamil_brahmi"
                   / "general_brahmi_letters")
        if ref_dir.exists():
            for i, f in enumerate(sorted(self._image_files(ref_dir)), 1):
                lexicon["tamil_brahmi_letters"][f"letter_{i:02d}"] = {
                    "reference_file": f.name,
                    "brahmi_char": "",        # e.g. "\U00011013" or 'ka'
                    "transliteration": "",    # e.g. "ka", "vi", "na"
                    "meaning": "",
                }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lexicon, f, ensure_ascii=False, indent=2)
        logger.info(f"Created lexicon skeleton at {path} "
                    f"({len(lexicon['indus_signs'])} Indus signs, "
                    f"{len(lexicon['tamil_brahmi_letters'])} Brahmi letters). "
                    f"Fill in meanings/transliterations from your sources.")
        return lexicon

    @staticmethod
    def _image_files(directory):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        return [f for f in Path(directory).iterdir() if f.suffix.lower() in exts]

    # ── glyph preparation ──────────────────────────────────────────────
    def _binarize(self, gray):
        """Glyph-white binary image with consistent polarity (mirrors
        ImageNormalizer so train/val/refs all match)."""
        h, w = gray.shape
        corners = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
        if float(gray[h // 2, w // 2]) < float(np.mean(corners)) - 15:
            gray = 255 - gray
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.count_nonzero(binary) > (h * w) // 2:
            binary = 255 - binary
        return binary

    def _crop_to_bbox(self, binary, x, y, bw, bh):
        """Square-padded crop of one component, resized to 64x64 binary."""
        h, w = binary.shape
        pad = max(2, int(max(bw, bh) * 0.08))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = binary[y0:y1, x0:x1]
        ch, cw = crop.shape
        if ch < 4 or cw < 4:
            return None
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.uint8)
        square[(side - ch) // 2:(side - ch) // 2 + ch,
               (side - cw) // 2:(side - cw) // 2 + cw] = crop
        return cv2.resize(square, (64, 64), interpolation=cv2.INTER_NEAREST)

    # ── segmentation (outline-aware) ───────────────────────────────────
    def segment_inscription(self, image_path):
        """
        Split a potsherd drawing into individual letter glyphs.

        The pot outline / sherd ends are detected (border-touching, spanning
        components) and accounted for: they are reported via the returned
        outline flag but NEVER passed to classification.  When letters are
        fused to the outline, the outline band is subtracted and the interior
        letters are recovered.

        Returns (segments, outline_present) where segments is a list of
        {bbox, glyph} dicts sorted in left-to-right reading order.
        """
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return [], False
        h, w = gray.shape
        binary = self._binarize(gray)
        # sever outline glued to the image edge
        binary[:3, :] = 0
        binary[-3:, :] = 0
        binary[:, :3] = 0
        binary[:, -3:] = 0

        # erode a copy to break thin junctions; components are detected on
        # the eroded copy, glyphs are cropped from the original binary so
        # strokes stay intact
        eroded = cv2.erode(binary, np.ones((2, 2), np.uint8), iterations=1)

        n, labels, cstats, _cent = cv2.connectedComponentsWithStats(eroded, 8)
        comps = []
        outline_present = False
        for i in range(1, n):
            x, y, bw, bh, area = cstats[i]
            if area < 25:
                continue                      # speckle
            spanning = (bw * bh) > 0.7 * (h * w)
            touches = (x <= 6 or y <= 6 or x + bw >= w - 6 or y + bh >= h - 6)
            if spanning or (touches and (bh > 0.7 * h or bw > 0.7 * w)):
                # pot outline (or outline fused with letters): accounted for,
                # excluded from classification, interior letters recovered
                outline_present = True
                comp_mask = ((labels == i) * 255).astype(np.uint8)
                comps.extend(self._recover_letters(comp_mask, binary, h, w))
                continue
            comps.append([int(x), int(y), int(bw), int(bh)])

        # split abnormally wide components (joined letters) at blank columns
        split = []
        for (x, y, bw, bh) in comps:
            if bw > 0.45 * w:
                split.extend(self._projection_split(binary, x, y, bw, bh))
            else:
                split.append([x, y, bw, bh])
        comps = split

        # merge components with overlapping / near x-ranges
        # (letter body + detached diacritic marks)
        comps.sort(key=lambda c: c[0])
        groups = []
        for x, y, bw, bh in comps:
            if groups:
                gx, gy, gw, gh = groups[-1]
                gap_tol = max(6, int(0.12 * gw))
                if x <= gx + gw + gap_tol:
                    nx1, ny1 = max(gx + gw, x + bw), max(gy + gh, y + bh)
                    gx1, gy1 = min(gx, x), min(gy, y)
                    groups[-1] = [gx1, gy1, nx1 - gx1, ny1 - gy1]
                    continue
            groups.append([x, y, bw, bh])

        segments = []
        for (x, y, bw, bh) in groups:
            glyph = self._crop_to_bbox(binary, x, y, bw, bh)
            if glyph is not None:
                segments.append({"bbox": (int(x), int(y), int(bw), int(bh)),
                                 "glyph": glyph})
        return segments, outline_present

    def _recover_letters(self, comp_mask, binary, h, w):
        """Subtract the outline band of a fused component and return the
        bounding boxes of the interior letters."""
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # band must be thicker than the outline stroke so no arc sliver
        # survives (slivers would otherwise merge with real letters)
        thickness = max(16, int(0.08 * min(h, w)))
        band = np.zeros_like(comp_mask)
        cv2.drawContours(band, contours, -1, 255, thickness)
        # only the fused component's own pixels: free-standing letters are
        # separate components and must not be double-counted here
        inner = cv2.bitwise_and(comp_mask, cv2.bitwise_not(band))

        # ruled-table inscriptions: remove long horizontal ruling lines so
        # the letters between them separate (letter horizontals are short
        # and survive the open)
        rule_len = max(25, w // 12)
        horiz = cv2.morphologyEx(inner, cv2.MORPH_OPEN,
                                 np.ones((1, rule_len), np.uint8))
        inner = cv2.subtract(inner, horiz)

        n, _l, cstats, _c = cv2.connectedComponentsWithStats(inner, 8)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = cstats[i]
            if area < 25:
                continue
            if (bw * bh) > 0.4 * (h * w):
                continue                      # leftover outline fragment
            touches = (x <= 6 or y <= 6 or x + bw >= w - 6 or y + bh >= h - 6)
            if touches and (bh > 0.6 * h or bw > 0.6 * w):
                continue                      # residual outline arc
            boxes.append([int(x), int(y), int(bw), int(bh)])
        return boxes

    @staticmethod
    def _projection_split(binary, x, y, bw, bh):
        """Split a wide component into letters at blank column valleys."""
        h, w = binary.shape
        region = binary[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
        cols = np.count_nonzero(region, axis=0)
        parts, start = [], None
        for c, v in enumerate(cols):
            if v > 0 and start is None:
                start = c
            elif v == 0 and start is not None:
                if c - start >= 4:
                    parts.append((start, c))
                start = None
        if start is not None and len(cols) - start >= 4:
            parts.append((start, len(cols)))
        if len(parts) <= 1:
            return [[x, y, bw, bh]]
        return [[x + s, y, e - s, bh] for (s, e) in parts]

    # ── identification (parallel, both scripts) ────────────────────────
    def _load_brahmi_references(self):
        refs = []
        ref_dir = (self.data_dir / "processed" / "val" / "tamil_brahmi"
                   / "general_brahmi_letters")
        if not ref_dir.exists():
            return refs
        for i, f in enumerate(sorted(self._image_files(ref_dir)), 1):
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            binary = self._binarize(gray)
            n, _l, cstats, _c = cv2.connectedComponentsWithStats(binary, 8)
            if n < 2:
                continue
            i_max = 1 + int(np.argmax(cstats[1:, 4]))
            x, y, bw, bh, _a = cstats[i_max]
            glyph = self._crop_to_bbox(binary, x, y, bw, bh)
            if glyph is not None:
                refs.append({"letter_id": f"letter_{i:02d}", "glyph": glyph})
        return refs

    def match_brahmi(self, glyph):
        """Template-match a letter glyph against the reference alphabet.

        Both sides are dilated before the IoU so that scans of different
        dimensions / stroke thickness compare fairly (size tolerance).
        Returns (letter_id, iou_score).
        """
        if not self.brahmi_refs:
            return None, 0.0
        k = np.ones((3, 3), np.uint8)
        g = cv2.dilate((glyph > 0).astype(np.uint8), k, iterations=1) > 0
        best_id, best_iou = None, -1.0
        for ref in self.brahmi_refs:
            r = cv2.dilate((ref["glyph"] > 0).astype(np.uint8), k,
                           iterations=1) > 0
            inter = np.count_nonzero(g & r)
            union = np.count_nonzero(g | r)
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_id, best_iou = ref["letter_id"], iou
        return best_id, float(best_iou)

    def indus_top3(self, glyph):
        """Read one 64x64 binary glyph with the Indus CNN (top-3 + lexicon
        meanings - the project's adopted working readings)."""
        if self.classifier is None:
            return []
        X = (glyph.astype(np.float32) / 255.0).reshape(1, 64, 64, 1)
        probs = self.classifier.predict(X)[0]
        top3 = []
        for idx in np.argsort(probs)[::-1][:3]:
            name = self.class_names[idx]
            entry = self.lexicon["indus_signs"].get(name, {})
            top3.append({"class": name, "prob": float(probs[idx]),
                         "meaning": entry.get("meaning", "")})
        return top3

    # ── decoding ───────────────────────────────────────────────────────
    def _brahmi_entry(self, letter_id):
        return self.lexicon["tamil_brahmi_letters"].get(letter_id, {})

    def decode_inscription(self, image_path):
        """Segment a potsherd and read every letter against BOTH scripts."""
        segments, outline_present = self.segment_inscription(image_path)
        brahmi_tokens, indus_tokens = [], []
        for seg in segments:
            letter_id, score = self.match_brahmi(seg["glyph"])
            entry = self._brahmi_entry(letter_id)
            translit = entry.get("transliteration", "")
            top3 = self.indus_top3(seg["glyph"])
            seg.update({"letter_id": letter_id, "score": score,
                        "transliteration": translit,
                        "meaning": entry.get("meaning", ""),
                        "indus_top3": top3})
            brahmi_tokens.append(translit if translit else f"[{letter_id}?]")
            indus_tokens.append(top3[0]["class"] if top3 else "?")
        return {"file": Path(image_path).name, "segments": segments,
                "outline_present": outline_present,
                "brahmi_reading": "-".join(brahmi_tokens),
                "indus_reading": " | ".join(indus_tokens)}

    def decode_graffiti(self, processed_image):
        """Parallel read of one single-symbol graffiti (64x64 float 0..1):
        Indus CNN top-3 AND best Tamil-Brahmi letter, with annotations."""
        result = {"indus_top3": [], "brahmi": (None, 0.0),
                  "brahmi_translit": "", "brahmi_meaning": ""}
        if self.classifier is not None:
            X = processed_image.reshape(1, 64, 64, 1)
            probs = self.classifier.predict(X)[0]
            for idx in np.argsort(probs)[::-1][:3]:
                name = self.class_names[idx]
                entry = self.lexicon["indus_signs"].get(name, {})
                result["indus_top3"].append(
                    {"class": name, "prob": float(probs[idx]),
                     "meaning": entry.get("meaning", "")})
        glyph = ((processed_image.reshape(64, 64) > 0.5) * 255).astype(np.uint8)
        letter_id, score = self.match_brahmi(glyph)
        entry = self._brahmi_entry(letter_id)
        result["brahmi"] = (letter_id, score)
        result["brahmi_translit"] = entry.get("transliteration", "")
        result["brahmi_meaning"] = entry.get("meaning", "")
        return result

    # ── NLP composition (multi-letter decode) ─────────────────────────
    @staticmethod
    def _skeleton(word):
        return "".join(ch for ch in word if ch not in "aeiou")

    def nlp_decode(self, result, corpus_key=""):
        """Compose a multi-letter CNN reading into a word/meaning (NLP step).

        1. Syllabify the letter transliterations (a following independent
           vowel lengthens the previous consonant syllable: ha+ii -> hii).
        2. Fuzzy-match the composed word against attested Keeladi names.
        3. Attach the corpus reading of the sherd when known (e.g. the
           Atan corpus reads 'Kuviran Atan').
        4. Compose the Indus top-1 sign meanings into a gloss phrase.
        """
        translits = [s.get("transliteration", "") for s in result["segments"]]
        syllables = []
        for t in translits:
            if not t:
                continue
            if (syllables and t in VOWEL_TOKENS
                    and syllables[-1] not in VOWEL_TOKENS):
                syllables[-1] = syllables[-1][:-1] + t   # ka+ii -> kii
            else:
                syllables.append(t)
        word = "".join(syllables)

        name, ratio = None, 0.0
        if word:
            for cand, _meaning in KEELADI_NAMES.items():
                r = max(difflib.SequenceMatcher(None, word, cand).ratio(),
                        difflib.SequenceMatcher(
                            None, self._skeleton(word),
                            self._skeleton(cand)).ratio())
                if r > ratio:
                    name, ratio = cand, r
        name_match = (name, float(ratio)) if ratio >= 0.6 else None

        corpus = CORPUS_READINGS.get(corpus_key)

        gloss_parts = []
        for seg in result["segments"]:
            if seg.get("indus_top3"):
                meaning = seg["indus_top3"][0].get("meaning", "")
                gloss_parts.append(meaning.split(". ")[0].rstrip(".")
                                   if meaning else seg["indus_top3"][0]["class"])
        return {"letters": translits, "syllables": syllables, "word": word,
                "name_match": name_match, "corpus": corpus,
                "indus_gloss": " + ".join(gloss_parts)}

    # ── visualization ──────────────────────────────────────────────────
    def visualize_decoding(self, image_path, result, output_path):
        """Annotated potsherd: each letter boxed with its parallel reads
        (Brahmi transliteration above, Indus top-1 below)."""
        img = cv2.imread(str(image_path))
        if img is None:
            return
        for seg in result["segments"]:
            x, y, bw, bh = seg["bbox"]
            brahmi = seg["transliteration"] or seg["letter_id"]
            indus = ""
            if seg.get("indus_top3"):
                indus = seg["indus_top3"][0]["class"].replace("sign_", "P")
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(img, f'B:{brahmi} ({seg["score"]:.2f})',
                        (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
            if indus:
                cv2.putText(img, f'I:{indus}', (x, y + bh + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 0), 1)
        cv2.imwrite(str(output_path), img)
