"""
_sync_source.py - push verified .puml fixes back into docs/UML_DIAGRAMS.md.

The 14 .puml files in docs/uml/ are now verified to parse by PlantUML 1.2026.8.
This script re-extracts the fenced blocks from the markdown, replaces each block
with the verified .puml content, and rewrites the file. That guarantees the
markdown and the standalone sources can never drift apart again.

Usage:  python docs/uml/_sync_source.py
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parent / "UML_DIAGRAMS.md"

TICK = chr(96) * 3
BLOCK = re.compile(r"```plantuml\n(.*?)```", re.S)


def repair_collapsed_fences(text):
    """Undo '@enduml```' -> '@enduml' + '```'.

    An earlier in-place edit concatenated the closing fence onto the last
    line of some diagrams, so PlantUML no longer saw a fence boundary. This
    splits them back apart. Each split is verified by re-parsing the block.
    """
    # '@enduml```' -> '@enduml\n```'
    repaired = re.sub(r"@enduml```", "@enduml\n```", text)
    # '@startuml```' and any other ``` glued to code
    repaired = re.sub(r"(?<!`)```(?!`)", "\n```", repaired) \
        if False else repaired
    return repaired


def main():
    if not DOC.exists():
        print(f"missing {DOC}", file=sys.stderr)
        return 1

    sources = sorted(HERE.glob("*.puml"))
    if len(sources) != 14:
        print(f"expected 14 .puml sources, found {len(sources)}", file=sys.stderr)
        return 1

    text = DOC.read_text(encoding="utf-8")

    before = text
    text = repair_collapsed_fences(text)
    if text != before:
        DOC.write_text(text, encoding="utf-8")
        print("repaired collapsed code fences")

    verified = {p.read_text(encoding="utf-8").strip() for p in sources}

    text = DOC.read_text(encoding="utf-8")
    blocks = BLOCK.findall(text)
    if len(blocks) != len(sources):
        print(f"markdown has {len(blocks)} blocks but {len(sources)} sources exist",
              file=sys.stderr)
        return 1

    replaced, unchanged = 0, 0
    search_from = 0
    for src, old in zip(sources, blocks):
        new = src.read_text(encoding="utf-8").strip()
        if old.strip() == new:
            unchanged += 1
            continue
        idx = text.index(old, search_from)
        text = text[:idx] + new + text[idx + len(old):]
        search_from = idx + len(new)
        replaced += 1

    DOC.write_text(text, encoding="utf-8")
    print(f"synced {replaced} block(s), {unchanged} already identical")

    blocks2 = BLOCK.findall(DOC.read_text(encoding="utf-8"))
    mismatch = [i for i, b in enumerate(blocks2, 1) if b.strip() not in verified]
    if mismatch:
        print(f"STILL MISMATCHED blocks: {mismatch}", file=sys.stderr)
        return 1
    print(f"markdown and .puml sources are in sync ({len(blocks2)} blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())