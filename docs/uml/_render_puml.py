"""
_render_puml.py - render every docs/uml/*.puml to a real image with PlantUML.

Talks to Java directly (`java -jar plantuml.jar -tpng ...`) rather than going
through the Python `plantuml` package: the JAR sits in this same folder, so the
file `plantuml.jar` shadows the `plantuml` module on sys.path and breaks the
package import. Driving java directly avoids that entirely.

Outputs land in docs/uml/images/ by default.

Usage:
    python docs/uml/_render_puml.py
    python docs/uml/_render_puml.py --format svg
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JAR = HERE / "plantuml.jar"

# PlantUML -t<fmt> codes
FMT = {"png": "png", "svg": "svg", "txt": "atxt", "pdf": "pdf"}


def render_one(jar, src, out_dir, fmt):
    """Render a single .puml. Returns (ok, message)."""
    cmd = [
        "java", "-jar", str(jar),
        "-t" + FMT[fmt],
        "-charset", "UTF-8",
        "-o", str(out_dir),
        str(src),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    dest = out_dir / src.with_suffix("." + fmt).name
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"{dest.name} ({dest.stat().st_size} bytes)"
    err = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
    return False, err[:200] or "no output produced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="png", choices=list(FMT))
    ap.add_argument("--out-dir", default=str(HERE / "images"))
    args = ap.parse_args()

    sources = sorted(HERE.glob("*.puml"))
    if not sources:
        print("no .puml sources found", file=sys.stderr)
        return 1

    if not JAR.exists():
        print(f"plantuml.jar not found at {JAR}", file=sys.stderr)
        print("Download from "
              "https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"format: {args.format}   out: {out_dir}")
    ok, failed = 0, []

    for src in sources:
        try:
            good, msg = render_one(JAR, src, out_dir, args.format)
        except subprocess.TimeoutExpired:
            good, msg = False, "timed out"
        if good:
            print(f"  [OK]   {src.name:42s} -> {msg}")
            ok += 1
        else:
            print(f"  [FAIL] {src.name:42s} -> {msg}")
            failed.append(src.name)

    print(f"\nrendered {ok}/{len(sources)} diagrams as {args.format}")
    if failed:
        print("failed:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())