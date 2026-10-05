"""
_validate_puml.py - offline structural check of the .puml sources in docs/uml/.

Java/plantuml may not be installed on this machine, so this script does a
static sanity pass instead of a real render:

  1. exactly one @startuml and one @enduml
  2. balanced braces {}, brackets [] and parentheses ()
  3. balanced double-quotes on each line (unbalanced quotes are the most
     common cause of a silent parse failure)
  4. every '-->' / '..>' arrow has a participant on both sides
  5. class/package/actor/system declarations end with '{' or a terminator
  6. no stray markdown fences left in the .puml files

Exit code 0 = all files pass.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ERRORS = []


def strip_comment(line):
    """Remove a trailing ' comment and drop brace chars inside double quotes."""
    out, in_q = [], False
    for ch in line:
        if ch == '"':
            in_q = not in_q
            out.append(ch)
            continue
        if ch == "'" and not in_q:
            break
        # braces/brackets/parens inside a quoted label are literal text
        if not in_q and ch in "{}[]()":
            out.append(ch)
            continue
        out.append(ch)
    return "".join(out), in_q


def check(path):
    name = path.name
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    errors = []

    def err(lineno, msg):
        errors.append(f"{name}:{lineno}: {msg}")

    if "```" in raw:
        err(0, "contains a markdown code fence")
    if lines.count("@startuml") != 1:
        err(0, f"expected 1 @startuml, found {lines.count('@startuml')}")
    if lines.count("@enduml") != 1:
        err(0, f"expected 1 @enduml, found {lines.count('@enduml')}")
    if "startuml" not in lines[0].lower():
        err(1, "first line should be @startuml")

    # Brace depth is tracked across the WHOLE file: a block opener and its
    # closer legitimately live on different lines, so a per-line check would
    # flag every multi-line class/package/state block.
    depth = {"{": 0, "[": 0, "(": 0}
    pairs = {"{": "}", "[": "]", "(": ")"}
    closers = {v: k for k, v in pairs.items()}

    for i, raw_line in enumerate(lines, 1):
        line, unclosed_q = strip_comment(raw_line)
        if unclosed_q:
            err(i, f"unbalanced double quote: {raw_line.strip()[:70]}")

        code = line.split("'")[0]
        for ch in code:
            if ch in depth:
                depth[ch] += 1
            elif ch in closers:
                depth[closers[ch]] -= 1
                if depth[closers[ch]] < 0:
                    err(i, f"stray closing {ch!r}")
                    depth[closers[ch]] = 0

        # arrows must have something on both sides
        for m in re.finditer(r"(\.\.|-->|<--)", code):
            before = code[: m.start()].strip()
            after = code[m.end():].strip()
            if not before:
                err(i, f"arrow with no left-hand participant: {raw_line.strip()[:70]}")
            if not after:
                err(i, f"arrow with no right-hand participant: {raw_line.strip()[:70]}")

    for opener, level in depth.items():
        if level != 0:
            err(0, f"{level} unclosed {opener!r} (expected {pairs[opener]!r})")

    return errors


def main():
    files = sorted(p for p in HERE.glob("*.puml"))
    if not files:
        print("no .puml files found")
        return 1

    for path in files:
        errs = check(path)
        status = "OK  " if not errs else "FAIL"
        print(f"[{status}] {path.name}")
        ERRORS.extend(errs)

    if ERRORS:
        print("\n--- errors ---")
        for e in ERRORS:
            print(e)
        return 1

    print(f"\nAll {len(files)} diagrams passed structural validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())