"""
_fix_puml.py - repair PlantUML sources so they actually parse.

Verified against PlantUML 1.2026.8 (java -jar plantuml.jar) with
`-checkonly`. Three constructs this build rejects were found empirically
(see _probe.py), and each fix below is verified the same way -- never applied
blind:

  1. `system "X" as Y` -- the C4 `system` keyword is NOT supported at all in
     this build (also rejected with `!pragma teoz` and with `!include <C4/...>`).
     Replaced with `rectangle`, which is the accepted equivalent for the
     system-boundary box in Figure 1.

  2. `participant RP as run_pipeline.main()` -- an UNQUOTED alias containing a
     dot or parentheses is a parse error. Quoting it ("run_pipeline.main()")
     parses fine.

  3. `as "A\\nB"` -- the \\n line-break escape inside a quoted alias is also a
     parse error. Replaced with a single space. (\\n is still fine in notes,
     activity labels and arrow labels, so those are left alone.)

Usage:  python docs/uml/_fix_puml.py
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JAR = HERE / "plantuml.jar"
SOURCES = sorted(HERE.glob("*.puml"))

# 1. C4 `system` element -> rectangle
RE_SYSTEM = re.compile(r'^(\s*)system\s+(".*?"|\S+)(.*)$')

# 2. unquoted alias holding a dot or parens: `X as run_pipeline.main()`
RE_UNQUOTED_ALIAS = re.compile(
    r'^(\s*(?:participant|actor|database|queue|boundary|control|entity)\s+\w+\s+as\s+)'
    r'([A-Za-z_][\w./()]*(?:\.[\w./()]+)*)'
    r'(\s*)$'
)

# 3. \n inside a quoted alias
RE_ALIAS_NL = re.compile(r'(as\s+")([^"]*?)(")')


def fix_line(line):
    out = line

    # --- fix 1: system -> rectangle -------------------------------------
    m = RE_SYSTEM.match(out)
    if m:
        out = f"{m.group(1)}rectangle {m.group(2)}{m.group(3)}"

    # --- fix 3: \n inside a quoted alias -> space -----------------------
    def flatten(mo):
        return mo.group(1) + re.sub(r"\s+", " ", mo.group(2).replace("\\n", " ")).strip() + mo.group(3)

    out = RE_ALIAS_NL.sub(flatten, out)

    # --- fix 2: quote an unquoted alias with dot/parens ------------------
    m = RE_UNQUOTED_ALIAS.match(out)
    if m and ("." in m.group(2) or "(" in m.group(2) or ")" in m.group(2)):
        out = f'{m.group(1)}"{m.group(2)}"{m.group(3)}'

    return out


def check(path):
    """(ok, message) from a real PlantUML parse check."""
    try:
        proc = subprocess.run(
            ["java", "-jar", str(JAR), "-checkonly", "-charset", "UTF-8", str(path)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    blob = ((proc.stdout or "") + (proc.stderr or "")).lower()
    if "no diagram found" in blob:
        return False, "no diagram found"
    if "error" in blob:
        return False, "syntax error"
    return True, "ok"


def main():
    if not JAR.exists():
        print(f"plantuml.jar missing at {JAR}", file=sys.stderr)
        return 2

    changed, bad = [], []

    for src in SOURCES:
        original = src.read_text(encoding="utf-8")
        fixed = "\n".join(fix_line(l) for l in original.splitlines()) + "\n"

        if fixed != original:
            src.write_text(fixed, encoding="utf-8")

        ok, msg = check(src)
        mark = "OK  " if ok else "FAIL"
        tag = " (repaired)" if fixed != original else ""
        print(f"  [{mark}] {src.name:42s} {msg}{tag}")
        if fixed != original:
            changed.append(src.name)
        if not ok:
            bad.append(src.name)

    print(f"\nrepaired {len(changed)} file(s)")
    if changed:
        print("  " + ", ".join(changed))
    if bad:
        print("\nSTILL FAILING:", file=sys.stderr)
        for n in bad:
            print("  " + n, file=sys.stderr)
        return 1
    print(f"All {len(SOURCES)} diagrams now parse cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())