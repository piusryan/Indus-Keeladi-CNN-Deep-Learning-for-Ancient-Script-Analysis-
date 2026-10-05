"""
process_model.py - report reader (de-fabricated).

This script used to write a report full of hard-coded "dummy but realistic
metrics" (accuracy 0.92, match_rate 75.5, high_conf = total * 0.60) and
np.random-generated scores.  That is deleted.

It now simply READS the real artefacts produced by the pipeline and prints
them.  If they do not exist yet, it tells you to run the pipeline instead of
inventing numbers.

Run:  python process_model.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_DIR = PROJECT_ROOT / "models" / "evaluation_results"


def _show(path, title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    path = Path(path)
    if not path.exists():
        print(f"  [missing] {path}\n  -> run: python run_pipeline.py")
        return None
    print(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def main():
    print("INDUS-KEELADI CNN - RESULTS READER (no generated numbers)")
    _show(EVAL_DIR / "validity_audit_report.txt", "VALIDITY AUDIT (leakage / open-set / verification)")
    _show(EVAL_DIR / "keeladi_evaluation_report.txt", "KEELADI MODEL EVALUATION")

    preds = EVAL_DIR / "keeladi_predictions.json"
    print("\n" + "=" * 74)
    print("PER-SHERD PREDICTIONS (model-derived)")
    print("=" * 74)
    if preds.exists():
        data = json.loads(preds.read_text(encoding="utf-8"))
        print(f"  generated: {data.get('generated')}  source: {data.get('source')}")
        for folder, rows in data.get("per_sherd", {}).items():
            print(f"\n  {folder} ({len(rows)} image(s))")
            for i, r in enumerate(rows[:5], 1):
                top = ", ".join(f"{c} {p:.3f}" for c, p in
                                zip(r["top3_classes"], r["top3_probs"]))
                print(f"    [{i}] {top}")
    else:
        print(f"  [missing] {preds}\n  -> run: python run_pipeline.py")


if __name__ == "__main__":
    main()
