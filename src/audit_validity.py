"""
Honest validity audit for the Indus-Keeladi CNN.

This module REPLACES the fabricated metrics that used to live in
``run_pipeline.py`` and ``process_model.py`` (``np.random.beta`` confidences,
hard-coded "75.5% match rate", estimated precision/recall) with three *real*,
reproducible experiments that any reviewer can re-run:

  1. LEAKAGE AUDIT      -- how much of the reported "validation accuracy" is
                           memorisation?  We measure nearest-neighbour image
                           duplication between train and val under (a) the
                           current random split and (b) a source-disjoint
                           (group) split where whole classes are held out.
  2. OPEN-SET AUDIT     -- can the closed-set 49-way softmax ever say
                           "this is not an Indus sign"?  We feed it known
                           negatives (noise/blank, Tamil-Brahmi letters,
                           general graffiti) and report the false-acceptance
                           rate against a positive control.
  3. VERIFICATION AUDIT -- for the 4 hand-picked Keeladi<->Indus pairs, does the
                           model really favour the expected sign?  We report
                           top-1/top-3 and a permutation-test p-value for
                           "better than chance".

Every number produced here is computed from the trained model and the images on
disk.  NOTHING is simulated.  If a number cannot be computed it is reported as
``null``, never invented.

Run:  python -m src.audit_validity      (or  python src/audit_validity.py)
"""

import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.image_normalization import ImageNormalizer

# Archaeological expectation map: Keeladi match folder -> ACCEPTABLE Indus classes.
# CORRECTED (verified against docs/*.pdf): the Keeladi annexure uses MAHADEVAN
# (M-1977) numbering, not the P-2010 numbers in the folder names.
#   * annexure "INDUS SIGN-225" = Fig.65 SERIAL 24 (M=225, P=219), not serial 25.
#   * annexure "INDUS SIGN-307" = Fig.65 SERIAL 18 (M="304 or 307", P="181 or 187").
# A folder maps to a LIST; matching ANY member counts as correct.
EXPECTED_MATCH_MAP = {
    "match_Indus_225": ["sign_24_P219"],
    "match_Indus_307": ["sign_18_P181_P187"],
    "match_Indus_318": ["sign_42_P318", "sign_42_P318b"],
    "match_Indus_365": ["sign_43_P365"],
}

# A folder with value None has no training class yet and cannot be scored.
def expected_classes(folder_name):
    """Return the list of acceptable class names for a match folder (always a list)."""
    v = EXPECTED_MATCH_MAP.get(folder_name)
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _image_files(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(f for f in directory.rglob("*")
                  if f.suffix.lower() in IMG_EXTS and f.is_file())


def load_class_names(model_path):
    """Class names are stored next to the .keras file as <stem>_classes.txt."""
    model_path = Path(model_path)
    names_path = model_path.parent / f"{model_path.stem}_classes.txt"
    return [ln.strip() for ln in names_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def load_classifier(model_path):
    from tensorflow import keras
    return keras.models.load_model(str(model_path))


def predict_probs(model, images):
    """images: (N,64,64) -> (N, num_classes) softmax array."""
    X = np.asarray(images, dtype=np.float32).reshape(-1, 64, 64, 1)
    return np.asarray(model.predict(X, verbose=0))


# ──────────────────────────────────────────────────────────────────────────
# 1. LEAKAGE AUDIT
# ──────────────────────────────────────────────────────────────────────────
def load_train_corpus(data_dir, class_names, normalizer, max_per_class=None):
    """
    Load the training corpus with an explicit *source group* per image.

    Source group = the canonical sign class the image was resolved to. In this
    repository every class folder holds one source drawing plus its augmented
    clones, so images in a folder share an origin. A split that puts images of
    the same origin on both sides of train/val therefore leaks.

    This mirrors src.train.load_training_data exactly: it applies the same
    allograph merge (ALLOGRAPH_GROUPS) and the same annexure remap
    (ANNEXURE_REMAP), so the audit measures the corpus the model was actually
    trained on.

    Returns (X, y, groups, class_names) where groups[i] is the source id.
    """
    from src.train import canonical_class, ANNEXURE_REMAP

    data_dir = Path(data_dir)
    search_dirs = [
        data_dir / "processed" / "train" / "primary_core_signs",
        data_dir / "processed" / "train" / "indus_matched",
    ]
    # canonical class -> [(dir, folder)] resolved exactly like the trainer
    canon_to_members = {}
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
            canon_to_members.setdefault(cls, []).append((sd, sub))

    X, y, groups = [], [], []
    for ci, cls in enumerate(class_names):
        members = canon_to_members.get(cls, [])
        imgs = []
        for _sd, sub in members:
            imgs.extend(_image_files(sub))
        if not imgs:
            continue
        if max_per_class:
            imgs = imgs[:max_per_class]
        for f in imgs:
            try:
                X.append(normalizer.process_image(f))
                y.append(ci)
                groups.append(cls)
            except Exception:
                continue
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(groups), class_names


def _unit(a):
    a = a.reshape(len(a), -1).astype(np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-8)


def _contamination(X, train_idx, val_idx, thresh):
    """Max cosine similarity of every val image to the train set."""
    A = _unit(X[train_idx])
    B = _unit(X[val_idx])
    sim = B @ A.T
    max_sim = sim.max(axis=1) if sim.size else np.zeros(len(val_idx))
    return {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "pct_val_with_near_duplicate": float(np.mean(max_sim >= thresh) * 100),
        "pct_val_exact_duplicate": float(np.mean(max_sim >= 0.9999) * 100),
        "mean_max_similarity": float(np.mean(max_sim)) if len(max_sim) else None,
        "median_max_similarity": float(np.median(max_sim)) if len(max_sim) else None,
    }


def leakage_audit(X, y, groups, test_size=0.15, thresh=0.99, seed=42):
    """
    Compare contamination under two protocols:
      random           : shuffle all images (what src/train.py does today)
      source_disjoint  : hold out whole classes (no shared origin in train)
    """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    idx = np.arange(len(X))

    tr_r, va_r = train_test_split(idx, test_size=test_size, random_state=seed)
    random_res = _contamination(X, tr_r, va_r, thresh)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_g, va_g = next(gss.split(X, y, groups=groups))
    group_res = _contamination(X, tr_g, va_g, thresh)

    return {
        "protocol_random_split": random_res,
        "protocol_source_disjoint_split": group_res,
        "similarity_threshold": thresh,
        "verdict": (
            "CONTAMINATED: val accuracy reflects memorisation of aug-clones"
            if random_res["pct_val_with_near_duplicate"] > 5 else "clean"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────
# 2. OPEN-SET AUDIT (false-acceptance rate)
# ──────────────────────────────────────────────────────────────────────────
def _negative_groups(normalizer, data_dir):
    """Build labelled negative groups of (64,64) images seen as non-signs."""
    rng = np.random.default_rng(0)
    groups = {}

    # obvious non-glyph negatives: blank, saturated, uniform noise
    groups["negative_blank"] = np.zeros((5, 64, 64), dtype=np.float32)
    groups["negative_saturated"] = np.ones((5, 64, 64), dtype=np.float32)
    groups["negative_noise"] = rng.random((20, 64, 64)).astype(np.float32)

    # real out-of-distribution marks: Tamil-Brahmi letters (letters, not IVS signs)
    tb = data_dir / "processed" / "val" / "tamil_brahmi" / "general_brahmi_letters"
    imgs = []
    for f in _image_files(tb):
        try:
            imgs.append(normalizer.process_image(f))
        except Exception:
            pass
    if imgs:
        groups["negative_tamil_brahmi_letters"] = np.asarray(imgs, dtype=np.float32)

    # general Keeladi graffiti (hypothesis: these *may* be signs)
    gen = data_dir / "processed" / "val" / "keeladi" / "general_keeladi_graffiti"
    imgs = []
    for f in _image_files(gen):
        try:
            imgs.append(normalizer.process_image(f))
        except Exception:
            pass
    if imgs:
        groups["general_keeladi_graffiti"] = np.asarray(imgs, dtype=np.float32)
    return groups


def openset_audit(model, class_names, normalizer, data_dir, n_pos=100, seed=42):
    """
    Report the top-1 softmax distribution for negatives vs a positive control.
    A closed-set model that scores negatives like positives is structurally
    unable to reject non-signs -> every 'match' is suspect.
    """
    rng = np.random.default_rng(seed)
    groups = _negative_groups(normalizer, data_dir)

    Xtr, _, _, _ = load_train_corpus(data_dir, class_names, normalizer)
    if len(Xtr) > n_pos:
        sel = rng.choice(len(Xtr), size=n_pos, replace=False)
        groups["positive_indus_train"] = Xtr[sel]
    elif len(Xtr):
        groups["positive_indus_train"] = Xtr

    results = {}
    for name, imgs in groups.items():
        imgs = np.asarray(imgs, dtype=np.float32)
        if imgs.ndim != 3 or len(imgs) == 0:
            continue
        probs = predict_probs(model, imgs)
        top1 = probs.max(axis=1)
        results[name] = {
            "n": int(len(imgs)),
            "mean_top1_confidence": float(np.mean(top1)),
            "pct_conf_ge_0.5": float(np.mean(top1 >= 0.5) * 100),
            "pct_conf_ge_0.8": float(np.mean(top1 >= 0.8) * 100),
        }

    neg = [v for k, v in results.items() if k.startswith("negative_")]
    if neg:
        mean_fa = float(np.mean([v["pct_conf_ge_0.5"] for v in neg]))
        results["_summary"] = {
            "mean_negative_confidence": float(np.mean([v["mean_top1_confidence"] for v in neg])),
            "mean_negative_false_accept_ge_0.5": mean_fa,
            "interpretation": (
                "Closed-set softmax accepts non-signs at high confidence "
                "(false-acceptance > 20%) -> 'match rate' cannot be trusted "
                "without an explicit open-set/negative control."
            ) if mean_fa > 20 else "Negatives are largely rejected; closed-set risk is low.",
        }
    return results



# ──────────────────────────────────────────────────────────────────────────
# 3. VERIFICATION AUDIT (do the hand-picked pairs beat chance?)
# ──────────────────────────────────────────────────────────────────────────
def verification_audit(model, class_names, normalizer, data_dir, n_perm=2000, seed=42):
    """
    For each hand-picked Keeladi<->Indus pair, report the real model verdict:
    is the EXPECTED sign actually favoured, and is the pairing better than a
    random pairing (permutation test)?
    """
    rng = np.random.default_rng(seed)
    val_dir = data_dir / "processed" / "val" / "keeladi"
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    pairs = []
    for folder in EXPECTED_MATCH_MAP:
        # Accept ANY of the expected classes (allographs / shared labels).
        present = [e for e in expected_classes(folder) if e in name_to_idx]
        if not present:
            continue
        files = _image_files(val_dir / folder)
        if not files:
            continue
        img = normalizer.process_image(files[0])
        probs = predict_probs(model, img[None, ...])[0]
        order = np.argsort(probs)[::-1]
        exp_idxs = [name_to_idx[e] for e in present]
        # probability assigned to the best acceptable class
        best_exp = max(exp_idxs, key=lambda i: probs[i])
        pairs.append({
            "folder": folder,
            "expected_class": "/".join(present),
            "expected_prob": float(probs[best_exp]),
            "top1_class": class_names[int(order[0])],
            "top1_prob": float(probs[order[0]]),
            "top3_classes": [class_names[int(i)] for i in order[:3]],
            "top3_probs": [float(probs[i]) for i in order[:3]],
            "top1_correct": bool(int(order[0]) in exp_idxs),
            "in_top3": bool(any(i in exp_idxs for i in order[:3])),
        })

    if not pairs:
        return {"pairs": [], "note": "no scorable pairs found"}

    true_mean = float(np.mean([p["expected_prob"] for p in pairs]))
    n_classes = len(class_names)
    # Proper permutation: recompute the model's prob for a random class per pair
    prob_matrix = np.stack([
        predict_probs(model, normalizer.process_image(
            _image_files(val_dir / p["folder"])[0])[None, ...])[0] for p in pairs
    ])
    null = np.array([
        float(np.mean([prob_matrix[i, rng.integers(0, n_classes)] for i in range(len(pairs))]))
        for _ in range(n_perm)
    ])
    p_value = float(np.mean(null >= true_mean))

    return {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_top1_correct": int(sum(p["top1_correct"] for p in pairs)),
        "n_in_top3": int(sum(p["in_top3"] for p in pairs)),
        "mean_expected_prob": true_mean,
        "null_mean": float(np.mean(null)),
        "null_p95": float(np.percentile(null, 95)),
        "permutation_p_value": p_value,
        "interpretation": (
            "The expected sign is NOT favoured above chance "
            "(p>=%.3f): the reported 'matches' are not reproduced by the CNN."
            % p_value
        ) if p_value >= 0.05 else
            "Expected signs are favoured above chance (p<0.05).",
    }



# ──────────────────────────────────────────────────────────────────────────
# Report + orchestration
# ──────────────────────────────────────────────────────────────────────────
def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_report(audit, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validity_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    L = []
    L.append("=" * 78)
    L.append("INDUS-KEELADI CNN - VALIDITY AUDIT (model-derived, nothing simulated)")
    L.append("=" * 78)
    L.append("")
    L.append("Every number below was computed by running the trained model over the")
    L.append("images on disk.  This file REPLACES the previous fabricated report.")
    L.append("")

    lk = audit.get("leakage_audit", {})
    L.append("-" * 78)
    L.append("1. LEAKAGE AUDIT  (is 'validation accuracy' real?)")
    L.append("-" * 78)
    for proto in ("protocol_random_split", "protocol_source_disjoint_split"):
        d = lk.get(proto, {})
        L.append(f"  {proto}:")
        L.append(f"     n_train={d.get('n_train')}  n_val={d.get('n_val')}")
        L.append(f"     % val with >= {lk.get('similarity_threshold')} twin in train: "
                 f"{_fmt(d.get('pct_val_with_near_duplicate'), 2)}%")
        L.append(f"     % val exact duplicate:  {_fmt(d.get('pct_val_exact_duplicate'), 2)}%")
        L.append(f"     mean max similarity:    {_fmt(d.get('mean_max_similarity'), 4)}")
    L.append(f"  VERDICT: {lk.get('verdict')}")
    L.append("")

    os_ = audit.get("openset_audit", {})
    L.append("-" * 78)
    L.append("2. OPEN-SET AUDIT  (can the model say 'not an Indus sign'?)")
    L.append("-" * 78)
    L.append(f"  {'group':38s} {'n':>4s} {'meanTop1':>9s} {'>=0.5%':>8s} {'>=0.8%':>8s}")
    for k, v in os_.items():
        if k.startswith("_"):
            continue
        L.append(f"  {k:38s} {v['n']:>4d} {v['mean_top1_confidence']:>9.4f} "
                 f"{v['pct_conf_ge_0.5']:>8.1f} {v['pct_conf_ge_0.8']:>8.1f}")
    if "_summary" in os_:
        L.append(f"  SUMMARY: {os_['_summary']['interpretation']}")
    L.append("")

    va = audit.get("verification_audit", {})
    L.append("-" * 78)
    L.append("3. VERIFICATION AUDIT  (are the 4 hand-picked pairs real?)")
    L.append("-" * 78)
    for p in va.get("pairs", []):
        L.append(f"  {p['folder']:16s} expected={p['expected_class']:22s}")
        L.append(f"      expected_prob={_fmt(p['expected_prob'])}  "
                 f"top1={p['top1_class']} ({_fmt(p['top1_prob'])})  "
                 f"top1_correct={p['top1_correct']}  in_top3={p['in_top3']}")
    if va.get("pairs"):
        L.append(f"  Top-1 correct: {va.get('n_top1_correct')}/{va.get('n_pairs')}   "
                 f"in top-3: {va.get('n_in_top3')}/{va.get('n_pairs')}")
        L.append(f"  mean P(expected sign)={_fmt(va.get('mean_expected_prob'))}  "
                 f"null_mean={_fmt(va.get('null_mean'))}  p={_fmt(va.get('permutation_p_value'), 4)}")
        L.append(f"  VERDICT: {va.get('interpretation')}")
    L.append("")
    L.append("-" * 78)
    L.append("CRITICAL FRAMING: this audit measures VISUAL similarity only. It cannot")
    L.append("establish script descent or decipherment. See docs/ANALYSIS_ACCURACY_ROADMAP.md")
    L.append("-" * 78)

    text = "\n".join(L)
    (out_dir / "validity_audit_report.txt").write_text(text, encoding="utf-8")
    return text


def run_audit(model_path=None, data_dir=None, out_dir=None, n_perm=2000):
    model_path = Path(model_path or PROJECT_ROOT / "models" / "indus_classifier.keras")
    data_dir = Path(data_dir or PROJECT_ROOT / "data")
    out_dir = Path(out_dir or PROJECT_ROOT / "models" / "evaluation_results")

    normalizer = ImageNormalizer(target_size=(64, 64))
    class_names = load_class_names(model_path)
    model = load_classifier(model_path)

    X, y, groups, _ = load_train_corpus(data_dir, class_names, normalizer)
    audit = {
        "model": str(model_path),
        "n_classes": len(class_names),
        "n_train_images": int(len(X)),
        "leakage_audit": leakage_audit(X, y, groups),
        "openset_audit": openset_audit(model, class_names, normalizer, data_dir),
        "verification_audit": verification_audit(model, class_names, normalizer, data_dir, n_perm=n_perm),
    }
    write_report(audit, out_dir)
    return audit


def main():
    audit = run_audit()
    print(write_report(audit, PROJECT_ROOT / "models" / "evaluation_results"))


if __name__ == "__main__":
    main()
