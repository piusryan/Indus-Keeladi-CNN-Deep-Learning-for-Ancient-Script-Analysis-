"""
siamese_embed.py - metric learning on the real Figure-65 sign glyphs.

The classifier in `src/train.py` is trained on synthetic clones of one drawing
per sign, so its embedding is not a true shape metric. Here we instead learn a
L2-normalised embedding with a batch-hard triplet loss + classification head,
using the 39 REAL glyphs digitised by `src/digitise_fig65.py` plus random
affine/noise perturbations of them.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
The held-out test perturbations are drawn from the same augmentation family as
the training ones. So the reported retrieval accuracy measures **robustness to
geometric perturbation of the same source glyph**, NOT generalisation to
genuine, independently-drawn allographs (we do not have those). It is an
upper bound on real-world performance, and is reported as such.

Run:  python -m src.siamese_embed
"""

import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import cv2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GLYPH_DIR = PROJECT_ROOT / "data" / "processed" / "train" / "fig65_real"
RESULTS_DIR = PROJECT_ROOT / "models" / "evaluation_results"
MODEL_OUT = PROJECT_ROOT / "models" / "fig65_siamese.keras"
EMB_OUT = RESULTS_DIR / "fig65_embeddings.json"
REPORT_OUT = RESULTS_DIR / "siamese_report.json"

IMG = 64
N_CLASSES = 39
EMBED_DIM = 64
# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_glyphs():
    """Load the digitised Fig.65 glyphs (ink=255 on black, as saved)."""
    man = json.loads((GLYPH_DIR / "manifest.json").read_text(encoding="utf-8"))
    items = []
    for k in sorted(man):
        img = cv2.imread(str(GLYPH_DIR / man[k]["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ink = (img < 128).astype(np.uint8) * 255      # stored inverted
        items.append((k, ink, man[k]))
    return items


def fit_canvas(img, size=IMG, margin=6):
    """Centre a glyph on a square black canvas, preserving aspect ratio."""
    h, w = img.shape[:2]
    s = (size - 2 * margin) / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def augment(img, rng):
    """Random affine + stroke-thickness + noise perturbation."""
    size = img.shape[0]
    ang = rng.uniform(-12, 12)
    sc = rng.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, sc)
    M[0, 2] += rng.uniform(-3, 3)
    M[1, 2] += rng.uniform(-3, 3)
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    r = rng.uniform(-1, 1)
    if r > 0.35:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))
    elif r < -0.35:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    if rng.random() < 0.5:
        out = np.clip(out.astype(np.float32)
                      + rng.normal(0, rng.uniform(4, 14), out.shape),
                      0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            out = (out > rng.uniform(60, 140)).astype(np.uint8) * 255
    return out


def build_dataset(items, per_class=48, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, (_k, ink, _meta) in enumerate(items):
        base = fit_canvas(ink)
        for _ in range(per_class):
            X.append(augment(base, rng).astype(np.float32) / 255.0)
            y.append(ci)
    return np.stack(X)[..., None], np.asarray(y, np.int32)


def build_test(items, seed=1234):
    """Held-out perturbations (different seed => different perturbation draws)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, (_k, ink, _meta) in enumerate(items):
        base = fit_canvas(ink)
        for _ in range(6):
            X.append(augment(base, rng).astype(np.float32) / 255.0)
            y.append(ci)
    return np.stack(X)[..., None], np.asarray(y, np.int32)


def build_reference(items):
    """Clean, un-augmented glyphs - the retrieval gallery."""
    X = np.stack([fit_canvas(ink) for _k, ink, _m in items]).astype(np.float32)
    return (X / 255.0)[..., None]
# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def make_model():
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = layers.Input((IMG, IMG, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    emb = layers.Dense(EMBED_DIM, name="embedding")(x)
    emb = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                        name="l2norm")(emb)
    logits = layers.Dense(N_CLASSES, name="logits")(emb)
    return Model(inp, [emb, logits], name="fig65_siamese")


def margin_triplet(emb, labels, margin=0.35):
    """Batch-hard triplet loss: hardest positive vs hardest negative in batch."""
    import tensorflow as tf
    d = tf.reduce_sum(tf.square(emb[:, None, :] - emb[None, :, :]), axis=2)
    d = tf.maximum(d, 0.0)
    same = tf.cast(tf.equal(labels[:, None], labels[None, :]), tf.float32)
    eye = tf.eye(tf.shape(labels)[0])
    pos_mask = same - eye                            # >0 iff same class, not self
    neg_mask = 1.0 - same
    # rows without a positive or without a negative would yield the 1e9 fill
    # value, so exclude them from the average instead of averaging them in.
    has_pos = tf.cast(tf.reduce_sum(pos_mask, axis=1) > 0, tf.float32)
    has_neg = tf.cast(tf.reduce_sum(neg_mask, axis=1) > 0, tf.float32)
    valid = has_pos * has_neg
    BIG = 1e6
    pos = tf.reduce_min(d + (1.0 - pos_mask) * BIG, axis=1)   # hardest positive
    neg = tf.reduce_max(d - (1.0 - neg_mask) * BIG, axis=1)   # hardest negative
    tri = tf.maximum(pos - neg + margin, 0.0)
    return tf.reduce_sum(tri * valid) / (tf.reduce_sum(valid) + 1e-9)


def encode(model, X, bs=128):
    return model.predict(X, batch_size=bs, verbose=0)[0]


# --------------------------------------------------------------------------- #
# train + evaluate
# --------------------------------------------------------------------------- #
def train(items, epochs=30, per_class=48, batch=96, seed=0):
    import tensorflow as tf
    Xtr, ytr = build_dataset(items, per_class, seed)
    Xte, yte = build_test(items)
    model = make_model()
    opt = tf.keras.optimizers.Adam(1e-3)
    rng = np.random.default_rng(seed + 7)
    steps = max(1, len(Xtr) // batch)
    n_cls = len(items)
    k = 6                                   # instances per class per batch
    bcls = min(n_cls, batch // k)           # classes per batch
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        by_cls = {c: perm[ytr[perm] == c] for c in range(n_cls)}
        tot = 0.0
        for i in range(steps):
            # class-balanced batch: bcls classes x k instances, so every anchor
            # has a positive as well as many hard negatives.
            cls = rng.choice(n_cls, size=bcls, replace=False)
            b = np.concatenate([rng.choice(by_cls[c], size=k, replace=False)
                                for c in cls])
            xb, yb = Xtr[b], ytr[b]
            with tf.GradientTape() as tape:
                emb, logits = model(xb, training=True)
                loss = (tf.reduce_mean(tf.keras.losses
                                      .sparse_categorical_crossentropy(
                                          yb, logits, from_logits=True))
                    + 2.0 * margin_triplet(emb, yb))
            g = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(g, model.trainable_variables))
            tot += float(loss)
        if ep % 2 == 0 or ep == epochs - 1:
            pred = model.predict(Xte, batch_size=128, verbose=0)[1].argmax(1)
            acc = float((pred == yte).mean())
            print(f"  epoch {ep:3d}  loss {tot / steps:.4f}  test-top1 {acc:.4f}",
                  flush=True)
    return model


def retrieval_report(model, items, Xte, yte):
    """Nearest-neighbour retrieval of held-out queries against the clean gallery."""
    ref = encode(model, build_reference(items))
    emb = encode(model, Xte)
    sim = emb @ ref.T                       # both L2-normalised -> cosine
    order = np.argsort(-sim, axis=1)
    ranks, top1, top5 = [], 0, 0
    for i in range(sim.shape[0]):
        r = int(np.where(order[i] == yte[i])[0][0]) + 1
        ranks.append(r)
        top1 += r == 1
        top5 += r <= 5
    n = len(ranks)
    ra = np.asarray(ranks)
    return {
        "n_query": n,
        "top1": round(top1 / n, 4),
        "top5": round(top5 / n, 4),
        "mean_rank": round(float(ra.mean()), 3),
        "median_rank": float(np.median(ra)),
        "chance_top1": round(1.0 / N_CLASSES, 4),
        "chance_mean_rank": round((N_CLASSES + 1) / 2.0, 2),
        "note": ("Held-out perturbations of the SAME source glyphs. Measures "
                 "robustness to geometric perturbation, NOT generalisation to "
                 "independently drawn allographs. Upper bound only."),
    }


def per_class_report(model, items, Xte, yte):
    """Top-1 per sign - shows which signs the embedding cannot separate.

    The simple stroke signs (Fig.65 serials 9-16 and 30) are expected to score
    near chance, because they differ only in stroke count/position. That is why
    they are excluded from match claims.
    """
    ref = encode(model, build_reference(items))
    emb = encode(model, Xte)
    order = np.argsort(-(emb @ ref.T), axis=1)
    per = {}
    for i in range(len(yte)):
        c = int(yte[i])
        ok = int(order[i][0]) == c
        d = per.setdefault(c, {"n": 0, "hit": 0, "stroke": False,
                               "key": items[c][0], "p2010": items[c][2]["p2010"]})
        d["n"] += 1
        d["hit"] += ok
    for c, d in per.items():
        d["top1"] = round(d["hit"] / d["n"], 3)
        del d["hit"], d["n"]
    out = {"per_class": {items[c][0]: d for c, d in sorted(per.items())}}
    stroke = [d["top1"] for c, d in per.items() if items[c][2].get("is_stroke_sign")]
    nonstroke = [d["top1"] for c, d in per.items()
                 if not items[c][2].get("is_stroke_sign")]
    out["stroke_sign_mean_top1"] = round(float(np.mean(stroke)), 3) if stroke else None
    out["nonstroke_mean_top1"] = round(float(np.mean(nonstroke)), 3) if nonstroke else None
    return out


def main(epochs=24, per_class=36, batch=72):
    items = load_glyphs()
    print(f"loaded {len(items)} real Fig.65 glyphs", flush=True)
    print(f"training Siamese (batch-hard triplet + CE), epochs={epochs}", flush=True)
    model = train(items, epochs=epochs, per_class=per_class, batch=batch)
    Xte, yte = build_test(items)
    rep = retrieval_report(model, items, Xte, yte)
    print("retrieval:", json.dumps({k: v for k, v in rep.items() if k != "note"},
                                   indent=2))
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    ref = encode(model, build_reference(items))
    EMB_OUT.write_text(json.dumps(
        {"keys": [k for k, _i, _m in items],
         "embeddings": [[round(float(v), 6) for v in e] for e in ref]},
        indent=2), encoding="utf-8")
    rep.update(per_class_report(model, items, Xte, yte))
    REPORT_OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("saved:", MODEL_OUT)


if __name__ == "__main__":
    main()