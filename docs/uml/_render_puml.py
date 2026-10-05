"""
_render_puml.py - render every docs/uml/*.puml to a real image using PlantUML.

Uses the local `plantuml` Python package. PlantUML needs a Java runtime and
either its bundled JAR or a downloaded one; this script locates the JAR, and
if it is absent prints exactly what to install rather than failing silently.

Outputs (default PNG) land next to each source in docs/uml/, or into
docs/uml/images/ with --out-dir.

Usage:
    python docs/uml/_render_puml.py
    python docs/uml/_render_puml.py --format svg --out-dir docs/uml/images
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# PlantUML jar filenames the Python package knows how to download/use.
JAR_NAMES = ("plantuml.jar", "PLANTUML.JAR")


def find_jar():
    """Locate plantuml.jar in the package dir, the env, or the system."""
    import plantuml

    candidates = [Path(plantuml.__file__).resolve().parent / n for n in JAR_NAMES]
    candidates += [HERE / n for n in JAR_NAMES]
    candidates += [Path("C:/plantuml.jar")]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="png",
                    choices=["png", "svg", "txt", "pdf"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    sources = sorted(HERE.glob("*.puml"))
    if not sources:
        print("no .puml sources found", file=sys.stderr)
        return 1

    jar = find_jar()
    if jar is None:
        print("plantuml.jar not found.", file=sys.stderr)
        print("Get it with either:", file=sys.stderr)
        print("  pip install plantuml   # then re-run this script", file=sys.stderr)
        print("  or download manually from "
              "https://github.com/plantuml/plantuml/releases", file=sys.stderr)
        return 2
    print(f"using jar: {jar}")

    from plantuml import PlantUML
    from plantuml.defines import OutputType

    out_type = {
        "png": OutputType.PNG,
        "svg": OutputType.SVG,
        "txt": OutputType.ATXT,
        "pdf": OutputType.PDF,
    }[args.format]

    out_dir = Path(args.out_dir) if args.out_dir else HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    server = PlantUML(url=jar, format=out_type, http_opts={"timeout": 60})
    ok, failed = 0, []

    for src in sources:
        dest = out_dir / src.with_suffix(f".{args.format}").name
        try:
            server.process_file(str(src), outfile=str(dest))
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  [OK]   {dest.name}  ({dest.stat().st_size} bytes)")
                ok += 1
            else:
                print(f"  [WARN] {src.name} produced no output")
                failed.append(src.name)
        except Exception as e:
            print(f"  [FAIL] {src.name}: {e}")
            failed.append(src.name)

    print(f"\nrendered {ok}/{len(sources)} diagrams as {args.format}")
    if failed:
        print("failed:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())