#!/usr/bin/env python3
"""
Secret Resizer – hidden per-atan red-box tuner
Usage:
  python secret_resizer.py --atan 3 --box 1 --scale 1.3 --dx -12 --dy 8
  python secret_resizer.py --atan 3 --box 2 --scale 0.9 --dx 5 --dy -4
  python secret_resizer.py --show
  python secret_resizer.py --reset --atan 3
  python secret_resizer.py --reset-all

Stores in .secret_resizer.json (hidden) which cnn_potsherd_annotator.py and run_pipeline.py auto-load.
"""
import json
import argparse
from pathlib import Path

CONFIG = Path(__file__).parent / ".secret_resizer.json"

def load():
    if CONFIG.exists():
        try:
            return json.loads(CONFIG.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

def save(data):
    CONFIG.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[secret] saved → {CONFIG} ({len(data)} atans)")

def main():
    p = argparse.ArgumentParser(description="Secret per-box resizer for atan red boxes")
    p.add_argument("--atan", type=int, help="atan number 1-10")
    p.add_argument("--box", type=int, help="box index 0-based (0,1,2...)")
    p.add_argument("--scale", type=float, help="scale factor 0.5-2.0 (e.g. 1.2 = 20% larger)")
    p.add_argument("--dx", type=int, default=0, help="x offset pixels")
    p.add_argument("--dy", type=int, default=0, help="y offset pixels")
    p.add_argument("--w", type=int, help="override width")
    p.add_argument("--h", type=int, help="override height")
    p.add_argument("--show", action="store_true", help="show current config")
    p.add_argument("--reset", action="store_true", help="reset one atan (needs --atan)")
    p.add_argument("--reset-all", action="store_true", help="reset all")
    args = p.parse_args()

    data = load()

    if args.show:
        print(json.dumps(data, indent=2) if data else "(empty) – no secret resizer active")
        return

    if args.reset_all:
        if CONFIG.exists():
            CONFIG.unlink()
        print("[secret] all reset – run_pipeline will use default boxes")
        return

    if args.reset and args.atan is not None:
        key = f"atan{args.atan}"
        if key in data:
            del data[key]
            save(data)
            print(f"[secret] reset {key}")
        else:
            print(f"[secret] {key} not in config")
        return

    if args.atan is not None and args.box is not None:
        key = f"atan{args.atan}"
        if key not in data:
            data[key] = {}
        box_key = str(args.box)
        entry = data[key].get(box_key, {})
        if args.scale is not None:
            entry["scale"] = args.scale
        if args.dx != 0:
            entry["dx"] = args.dx
        if args.dy != 0:
            entry["dy"] = args.dy
        if args.w is not None:
            entry["w"] = args.w
        if args.h is not None:
            entry["h"] = args.h
        # also allow setting via scale alone
        if args.scale is not None and args.dx == 0 and args.dy == 0 and args.w is None and args.h is None:
            entry["scale"] = args.scale
        data[key][box_key] = entry
        save(data)
        print(f"[secret] {key}[{box_key}] → {entry} – run `python run_pipeline.py` to apply")
        return

    p.print_help()
    print("\nExamples:")
    print("  python secret_resizer.py --atan 3 --box 1 --scale 1.4 --dx -15 --dy 10  # fix atan3 middle empty box")
    print("  python secret_resizer.py --atan 3 --box 0 --scale 0.85 --dx 8           # shrink left LLa that grabs border")
    print("  python secret_resizer.py --show")
    print("  python secret_resizer.py --reset --atan 3")

if __name__ == "__main__":
    main()
