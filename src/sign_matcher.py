"""
sign_matcher.py - open-set VERIFICATION / retrieval matcher for Indus<->Keeladi.

WHY THIS EXISTS
---------------
The CNN path in src/evaluate.py answers a *closed-set classification* question:
"which of the 43 classes is this image?" Its softmax output is therefore NOT a
similarity measure, and on Keeladi sherd photos it lands around 0.40-0.50 even
for genuinely paired pairs. Two further problems make it worse:

  1. A Keeladi "match" file is a *potsherd photograph*: it contains the pot
     outline and rim line as well as the incised glyph. The shared
     ImageNormalizer auto-crops to the LARGEST contour, which is usually the pot
     outline, not the glyph.
  2. The catalogue glyphs are thick, clean strokes; the sherd glyphs are thin
     scratches. A 43-way softmax is the wrong instrument for "is this the SAME
     SIGN as X?" - that is a *verification / retrieval* question.

This module therefore scores a sherd by:
  (a) segmenting it into individual glyphs (reusing InscriptionDecoder, which
      already excludes the outline),
  (b) CNN *embedding* cosine similarity to a per-class prototype, and
  (c) dilation-tolerant bidirectional stroke coverage against the class glyphs,
  (d) reporting the expected sign's RANK plus a calibrated percentage, and a
      permutation test for "better than chance".

Everything reported is computed from the images and the trained model. Nothing
is tuned per-pair and no score is hard-coded: if the morphology does not agree,
the rank stays low.

Run:  python -m src.sign_matcher
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.preprocessing.image_normalization import ImageNormalizer
from src.decoding import InscriptionDecoder
from src.audit_validity import (EXPECTED_MATCH_MAP, expected_classes,
                                load_class_names, load_classifier)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
log = logging.getLogger("sign_matcher")


def _image_files(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(f for f in directory.rglob("*")
                  if f.suffix.lower() in IMG_EXTS and f.is_file())


# ── mask helpers (stroke-level geometry) ───────────────────────────────────
def binarize(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(binary) > binary.size // 2:
        binary = 255 - binary
    return binary


def largest_component_mask(binary, size=64):
    """Largest connected component, cropped and padded to a square, then resized."""
    n, _lab, cstats, _cent = cv2.connectedComponentsWithStats(binary, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(cstats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _a = cstats[i]
    crop = binary[y:y + h, x:x + w]
    ch, cw = crop.shape
    side = max(ch, cw)
    square = np.zeros((side, side), dtype=np.uint8)
    square[(side - ch) // 2:(side - ch) // 2 + ch,
           (side - cw) // 2:(side - cw) // 2 + cw] = crop
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_NEAREST)


def stroke_coverage(a, b, dilate_iters=2):
    """Symmetric, dilation-tolerant stroke overlap in percent (0-100).

    How much of each drawing's strokes are present in the (dilated) other one.
    Robust to stroke-thickness differences between catalogue glyphs and
    scratched sherd glyphs.
    """
    A = (np.asarray(a) > 0).astype(np.uint8)
    B = (np.asarray(b) > 0).astype(np.uint8)
    na, nb = np.count_nonzero(A), np.count_nonzero(B)
    if not na or not nb:
        return 0.0
    k = np.ones((3, 3), np.uint8)
    Ad = cv2.dilate(A, k, iterations=dilate_iters)
    Bd = cv2.dilate(B, k, iterations=dilate_iters)
    ca = np.count_nonzero(A & Bd) / na
    cb = np.count_nonzero(B & Ad) / nb
    return 100.0 * (ca + cb) / 2.0


def embed_model(model):
    """Sub-model exposing the 128-d penultimate embedding (not the softmax).

    A Sequential loaded from a .keras file does not expose `.input` in Keras 3,
    so we replay the loaded layers over a fresh Input to reach the 128-d layer.
    """
    from tensorflow import keras
    inp = keras.Input(shape=(64, 64, 1))
    x = inp
    target = None
    for layer in model.layers:
        x = layer(x)
        units = getattr(layer, "units", None)
        if units and int(units) == 128:
            target = x
    if target is None:
        target = x            # fall back to the last layer
    return keras.Model(inp, target)


def embed_images(embedder, images):
    X = np.asarray(images, dtype=np.float32).reshape(-1, 64, 64, 1)
    E = np.asarray(embedder.predict(X, verbose=0))
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)
# ── class references (masks + images) ─────────────────────────────────────
def build_class_references(data_dir, class_names, limit=6):
    """class name -> {"masks": [...], "imgs": [...]} using the same canonical
    class resolution as the trainer (allograph merge + annexure remap)."""
    from src.train import canonical_class, ANNEXURE_REMAP

    data_dir = Path(data_dir)
    search_dirs = [
        data_dir / "processed" / "train" / "primary_core_signs",
        data_dir / "processed" / "train" / "indus_matched",
    ]
    canon_to_files = {}
    for sd in search_dirs:
        if not sd.exists():
            continue
        is_matched = (sd.name == "indus_matched")
        for sub in sorted(sd.iterdir()):
            if not sub.is_dir():
                continue
            cls = canonical_class(sub.name)
            if is_matched and sub.name in ANNEXURE_REMAP:
                cls = ANNEXURE_REMAP[sub.name]
            canon_to_files.setdefault(cls, []).extend(_image_files(sub))

    refs = {}
    for cls in class_names:
        files = canon_to_files.get(cls, [])
        # prefer the original drawing (not the synthetic aug_ clones)
        originals = [f for f in files if not f.stem.lower().startswith("aug")]
        chosen = (originals + files)[:limit]
        masks, imgs = [], []
        for f in chosen:
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            mask = largest_component_mask(binarize(gray))
            if mask is None:
                continue
            masks.append(mask)
            imgs.append(mask.astype(np.float32) / 255.0)
        if masks:
            refs[cls] = {"masks": masks, "imgs": imgs}
    return refs


def prototype_embeddings(embedder, refs):
    """class -> L2-normalised mean embedding over its reference glyphs."""
    protos = {}
    for cls, d in refs.items():
        E = embed_images(embedder, d["imgs"])
        m = E.mean(axis=0)
        protos[cls] = m / max(np.linalg.norm(m), 1e-8)
    return protos


# ── sherd glyph extraction ─────────────────────────────────────────────────
def extract_glyphs(decoder, sherd_path):
    """Per-letter glyph masks of a potsherd, pot outline excluded."""
    try:
        segments, _outline = decoder.segment_inscription(str(sherd_path))
    except Exception as e:
        log.debug(f"segmentation failed for {sherd_path}: {e}")
        return []
    glyphs = []
    for seg in segments:
        g = seg.get("glyph")
        if g is None:
            continue
        a = np.asarray(g)
        a = (a * 255).astype(np.uint8) if a.dtype != np.uint8 else a
        mask = largest_component_mask(binarize(a))
        if mask is not None and np.count_nonzero(mask) > 0:
            glyphs.append(mask)
    return glyphs


def score_sherd(glyph_masks, embedder, protos, refs, class_names):
    """Combined per-class score for one sherd.

    Each class keeps its BEST-matching glyph, so extra strokes on the sherd
    cannot dilute the score.
    """
    if not glyph_masks:
        return {c: 0.0 for c in class_names}, None

    E = embed_images(embedder, [m.astype(np.float32) / 255.0 for m in glyph_masks])
    cnn = np.zeros(len(class_names))
    for ci, cls in enumerate(class_names):
        p = protos.get(cls)
        cnn[ci] = float(np.max(E @ p)) if p is not None else 0.0
    lo, hi = float(cnn.min()), float(cnn.max())
    norm = (cnn - lo) / (hi - lo) if hi > lo else np.zeros_like(cnn)

    stroke = np.zeros(len(class_names))
    for gi, mask in enumerate(glyph_masks):
        for ci, cls in enumerate(class_names):
            d = refs.get(cls)
            if not d:
                continue
            best = max(stroke_coverage(mask, rm) for rm in d["masks"])
            if best > stroke[ci]:
                stroke[ci] = best

    combined = 100.0 * (0.5 * norm + 0.5 * (stroke / 100.0))
    return ({class_names[i]: float(combined[i]) for i in range(len(class_names))},
            combined)
# ── verification driver ────────────────────────────────────────────────────
def match_all(model_path=None, data_dir=None, n_perm=2000, seed=42):
    model_path = Path(model_path or PROJECT_ROOT / "models" / "indus_classifier.keras")
    data_dir = Path(data_dir or PROJECT_ROOT / "data")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    normalizer = ImageNormalizer(target_size=(64, 64))
    class_names = load_class_names(model_path)
    model = load_classifier(model_path)
    embedder = embed_model(model)
    decoder = InscriptionDecoder(data_dir, normalizer, classifier=model,
                                 class_names=class_names)

    log.info(f"classes={len(class_names)}  building class references...")
    refs = build_class_references(data_dir, class_names)
    usable = [c for c in class_names if c in refs]
    log.info(f"classes with usable references: {len(usable)}")
    protos = prototype_embeddings(embedder, refs)

    val_keeladi = data_dir / "processed" / "val" / "keeladi"
    rng = np.random.default_rng(seed)
    rows = []
    for folder in EXPECTED_MATCH_MAP:
        files = _image_files(val_keeladi / folder)
        if not files:
            continue
        sherd = files[0]
        glyphs = extract_glyphs(decoder, sherd)
        scores, vec = score_sherd(glyphs, embedder, protos, refs, class_names)
        if vec is None:
            rows.append({"folder": folder, "error": "no glyphs segmented"})
            continue
        order = np.argsort(vec)[::-1]
        exp = [e for e in expected_classes(folder) if e in scores]
        exp_scores = {e: scores[e] for e in exp}
        best_exp = max(exp, key=lambda e: scores[e]) if exp else None
        rank = (int(np.where(order == class_names.index(best_exp))[0][0]) + 1
                if best_exp else None)
        rows.append({
            "folder": folder,
            "n_glyphs": len(glyphs),
            "expected": exp,
            "expected_scores": exp_scores,
            "best_expected": best_exp,
            "expected_rank": rank,
            "expected_score_pct": float(scores[best_exp]) if best_exp else None,
            "top5": [{"class": class_names[int(i)],
                      "score": float(vec[int(i)])} for i in order[:5]],
            "in_top5": bool(best_exp in [class_names[int(i)] for i in order[:5]]),
            "_vec": vec,
        })

    # permutation test: is the expected sign favoured above a random sign?
    usable_rows = [r for r in rows if "_vec" in r]
    p_value = None
    if usable_rows:
        true_mean = float(np.mean([r["expected_score_pct"] for r in usable_rows]))
        null = np.empty(n_perm)
        for k in range(n_perm):
            picks = rng.integers(0, len(class_names), size=len(usable_rows))
            null[k] = np.mean([usable_rows[i]["_vec"][picks[i]]
                               for i in range(len(usable_rows))])
        p_value = float(np.mean(null >= true_mean))
        mean_true = true_mean
    else:
        mean_true = None

    for r in rows:
        r.pop("_vec", None)

    return {
        "model": str(model_path),
        "n_classes": len(class_names),
        "n_classes_with_refs": len(usable),
        "sherds": rows,
        "n_expected_in_top5": int(sum(r.get("in_top5", False) for r in rows)),
        "mean_expected_score_pct": mean_true,
        "null_mean": float(null.mean()) if usable_rows else None,
        "permutation_p_value": p_value,
    }


def write_report(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sign_match_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["=" * 78,
         "INDUS-KEELADI SIGN MATCHER - open-set verification (model-derived)",
         "=" * 78,
         "",
         "Each Keeladi sherd is SEGMENTED into glyphs (pot outline excluded) and",
         "scored against every Indus class by CNN embedding similarity +",
         "dilation-tolerant stroke coverage. Scores are computed, not tuned.",
         "",
         "-" * 78,
         f"{'folder':20s} {'glyphs':>6s} {'rank':>5s} {'score%':>7s}  top-1",
         "-" * 78]
    for r in result["sherds"]:
        if "error" in r:
            L.append(f"{r['folder']:20s}  ERROR: {r['error']}")
            continue
        top1 = r["top5"][0]
        L.append(f"{r['folder']:20s} {r['n_glyphs']:>6d} "
                 f"{str(r['expected_rank']):>5s} "
                 f"{(r['expected_score_pct'] or 0):>7.2f}  {top1['class']}")
        for i, t in enumerate(r["top5"], 1):
            mark = " <-- expected" if t["class"] in r["expected"] else ""
            L.append(f"      {i}. {t['class']:28s} {t['score']:6.2f}{mark}")
    L += ["-" * 78,
          f"expected sign in top-5 : {result['n_expected_in_top5']}/{len(result['sherds'])}",
          f"mean expected score%   : "
          f"{result['mean_expected_score_pct']:.2f}"
          if result["mean_expected_score_pct"] is not None else "n/a",
          f"random-class mean%     : "
          f"{result['null_mean']:.2f}" if result["null_mean"] is not None else "n/a",
          f"permutation p-value    : "
          f"{result['permutation_p_value']:.4f}"
          if result["permutation_p_value"] is not None else "n/a",
          "-" * 78,
          "NOTE: a high score means the SHAPES agree. It is not evidence of",
          "script descent; see docs/ANALYSIS_ACCURACY_ROADMAP.md.",
          ]
    text = "\n".join(L)
    (out_dir / "sign_match_report.txt").write_text(text, encoding="utf-8")
    return text


def main():
    result = match_all()
    print(write_report(result, PROJECT_ROOT / "models" / "evaluation_results"))


if __name__ == "__main__":
    main()