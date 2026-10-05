"""
_bisect_puml.py - find the exact offending line in a .puml file.

java -jar plantuml.jar -checkonly only says "Some diagram description contains
errors" without a line number. This binary-searches by rendering prefixes of
the file: a prefix that fails is narrowed further, so the first line that
introduces the error is isolated.

Usage:  python docs/uml/_bisect_puml.py docs/uml/06_sequence_pipeline.puml
"""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
JAR = HERE / "plantuml.jar"


def parses(text, tmpdir, idx):
    """Write `text` to a temp .puml and ask PlantUML to check it."""
    f = Path(tmpdir) / f"probe_{idx}.puml"
    f.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        ["java", "-jar", str(JAR), "-checkonly", "-charset", "UTF-8", str(f)],
        capture_output=True, text=True, timeout=120,
    )
    blob = (proc.stdout or "") + (proc.stderr or "")
    return "contains errors" not in blob and "error" not in blob.lower()


def bisect(lines, tmpdir):
    """Return the 1-based index of the first line that breaks the parse."""
    lo, hi = 0, len(lines)
    # confirm the full file fails
    if parses("\n".join(lines), tmpdir, 0):
        return None

    # shrink from the front: find the largest prefix that still fails
    while lo < hi:
        mid = (lo + hi) // 2
        chunk = lines[:mid]
        # ensure the chunk is a complete @startuml/@enduml document
        joined = "\n".join(chunk)
        if not joined.strip().startswith("@startuml"):
            lo = mid + 1
            continue
        if not joined.rstrip().endswith("@enduml"):
            lo = mid + 1
            continue
        if parses(joined, tmpdir, lo):
            lo = mid + 1
        else:
            hi = mid
    return hi  # lines[hi-1] is the culprit


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if not JAR.exists():
        print(f"plantuml.jar missing at {JAR}", file=sys.stderr)
        return 2

    for arg in sys.argv[1:]:
        p = Path(arg)
        lines = p.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as td:
            idx = bisect(lines, td)
        print(f"\n=== {p.name} ===")
        if idx is None:
            print("  parses cleanly")
            continue
        print(f"  first failing line: {idx}")
        lo = max(0, idx - 4)
        hi = min(len(lines), idx + 2)
        for j in range(lo, hi):
            mark = ">>>" if j + 1 == idx else "   "
            print(f"  {mark} {j + 1:4d}: {lines[j]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())