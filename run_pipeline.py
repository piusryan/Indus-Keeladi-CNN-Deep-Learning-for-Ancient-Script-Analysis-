"""
Honest end-to-end pipeline for the Indus-Keeladi CNN project.

The previous version of this script SIMULATED its results: it built training
curves with ``np.random.normal``, drew confidences from ``np.random.beta``,
picked sign labels with ``np.random.choice``, wrote ``np.random.uniform``
"known pair" scores, and printed a hard-coded "75.5% match rate".  All of that
has been removed.

Every number this script reports now comes from running the real code:

    * src.audit_validity  -> leakage / open-set / verification audit
    * src.evaluate        -> real per-sherd Keeladi predictions from the model
    * models/indus_classifier.keras -> the trained weights

If the model or the data is missing, the script STOPS and says so.  It never
invents a result.

Run:  python run_pipeline.py
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "indus_classifier.keras"
EVAL_DIR = MODEL_DIR / "evaluation_results"
DECODED_DIR = EVAL_DIR / "decoded"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pipeline")


def banner(text):
    log.info("\n" + "=" * 78)
    log.info(text)
    log.info("=" * 78)


def scan_data():
    """Count what is actually on disk (no fabrication, just file counts)."""
    counts = {}
    trains = [
        DATA_DIR / "processed" / "train" / "primary_core_signs",
        DATA_DIR / "processed" / "train" / "indus_matched",
        DATA_DIR / "processed" / "train" / "permanent_modifiers",
        DATA_DIR / "processed" / "train" / "diacritical_marks",
    ]
    counts["train_classes"] = sum(
        len([d for d in t.iterdir() if d.is_dir()]) for t in trains if t.exists())
    val_dirs = {
        "keeladi_match": DATA_DIR / "processed" / "val" / "keeladi",
        "tamil_brahmi": DATA_DIR / "processed" / "val" / "tamil_brahmi",
    }
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    total = 0
    for name, base in val_dirs.items():
        n = sum(1 for f in base.rglob("*") if f.suffix.lower() in exts) if base.exists() else 0
        counts[name] = n
        total += n
    counts["val_total"] = total
    return counts


def step_audit():
    """Validity audit: leakage, open-set false acceptance, verification."""
    from src.audit_validity import run_audit
    return run_audit(model_path=MODEL_PATH, data_dir=DATA_DIR, out_dir=EVAL_DIR)


def step_evaluate():
    """Real Keeladi evaluation from the trained model."""
    from src.evaluate import KeeladiEvaluator
    evaluator = KeeladiEvaluator(MODEL_PATH, DATA_DIR)
    validation_data = evaluator.load_keeladi_validation_set()
    if not validation_data:
        raise RuntimeError("No Keeladi/Tamil-Brahmi validation images found.")
    predictions = evaluator.predict_keeladi_matches(validation_data, threshold=0.5)
    analysis = evaluator.analyze_civilization_link(predictions)
    evaluator.generate_report(predictions, analysis, EVAL_DIR)
    try:
        evaluator.run_decoding(EVAL_DIR)
    except Exception as e:  # decoding is optional; never fake its output
        log.warning(f"  decoding step skipped: {e}")
    try:
        evaluator.generate_known_pair_comparison(EVAL_DIR)
    except Exception as e:
        log.warning(f"  known-pair comparison skipped: {e}")
    return predictions, analysis


def step_sign_match():
    """Open-set verification / retrieval matcher (segment + embedding + stroke)."""
    from src.sign_matcher import match_all, write_report
    result = match_all(model_path=MODEL_PATH, data_dir=DATA_DIR)
    text = write_report(result, EVAL_DIR)
    return result, text


def write_predictions_json(predictions, analysis):
    """Persist the REAL per-sherd top-3 predictions so the dashboard can show
    model output instead of a hard-coded table."""
    out = {}
    for folder, p in predictions.items():
        out[folder] = [
            {"top3_classes": list(p["top3_class_names"][i]),
             "top3_probs": [float(x) for x in p["top3_probs"][i]]}
            for i in range(len(p["all_predictions"]))
        ]
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "src/evaluate.KeeladiEvaluator (model-derived, not simulated)",
        "analysis": {k: v for k, v in analysis.items() if k != "direct_matches"},
        "per_sherd": out,
    }
    path = EVAL_DIR / "keeladi_predictions.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def step_annotate():
    """Real CNN potsherd annotation (optional)."""
    try:
        from cnn_potsherd_annotator import PotsherdAnnotator
        PotsherdAnnotator(DATA_DIR / "processed", DECODED_DIR).annotate_all()
        return True
    except Exception as e:
        log.warning(f"  potsherd annotation skipped: {e}")
        return False


def main():
    banner("INDUS-KEELADI CNN - HONEST PIPELINE (no simulated metrics)")
    log.info(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not MODEL_PATH.exists():
        log.error(f"Model not found: {MODEL_PATH}\n"
                  "Train a model first (python src/train.py). Refusing to fake results.")
        sys.exit(1)

    log.info("\n[1/6] Scanning data (real file counts)")
    counts = scan_data()
    for k, v in counts.items():
        log.info(f"   {k}: {v}")

    log.info("\n[2/6] Validity audit (leakage / open-set / verification)")
    audit = step_audit()
    lk = audit["leakage_audit"]["protocol_random_split"]
    log.info(f"   leakage: {lk['pct_val_with_near_duplicate']:.2f}% of val images "
             f"have a >=0.99 twin in train (random split)")
    va = audit["verification_audit"]
    log.info(f"   verification: {va.get('n_top1_correct')}/{va.get('n_pairs')} hand-picked "
             f"pairs predicted correctly (p={va.get('permutation_p_value'):.3f})")

    log.info("\n[3/6] Real Keeladi evaluation from the trained model")
    predictions, analysis = step_evaluate()
    log.info(f"   images analysed: {analysis['total_keeladi_images']}")
    log.info(f"   mean confidence: {analysis['mean_confidence']:.4f}")
    log.info(f"   high-confidence matches (>=0.5): {analysis['total_high_confidence_matches']}")
    log.info(f"   match rate: {analysis['match_rate']:.2%}")
    log.info(f"   known-pair top-1 correct: {analysis.get('known_pair_correct')}"
             f"/{analysis.get('known_pair_total')}")

    log.info("\n[4/6] Sign matching (open-set verification, segmented glyphs)")
    sm, _ = step_sign_match()
    for r in sm["sherds"]:
        if "error" in r:
            continue
        log.info(f"   {r['folder']:20s} score={r['expected_score_pct']:6.2f}%  "
                 f"rank={r['expected_rank']:>3}  top1={r['top5'][0]['class']}")
    log.info(f"   expected sign in top-5: {sm['n_expected_in_top5']}/{len(sm['sherds'])}"
             f" | mean score {sm['mean_expected_score_pct']:.2f}%"
             f" vs random {sm['null_mean']:.2f}%  p={sm['permutation_p_value']:.4f}")

    log.info("\n[5/6] Writing model-derived predictions for the dashboard")
    p = write_predictions_json(predictions, analysis)
    log.info(f"   {p}")

    log.info("\n[6/6] Annotating potsherds (real CNN/template output)")
    step_annotate()

    banner("PIPELINE COMPLETE")
    log.info(f"Audit report : {EVAL_DIR / 'validity_audit_report.txt'}")
    log.info(f"Sign match   : {EVAL_DIR / 'sign_match_report.txt'}")
    log.info(f"Eval report  : {EVAL_DIR / 'keeladi_evaluation_report.txt'}")
    log.info(f"Predictions  : {EVAL_DIR / 'keeladi_predictions.json'}")
    log.info("\nNOTE: reported similarity is VISUAL only; it cannot establish")
    log.info("script descent or decipherment. See docs/ANALYSIS_ACCURACY_ROADMAP.md")


if __name__ == "__main__":
    main()

