"""Structural check of the merged root DOCUMENTATION.md."""
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
p = ROOT / "DOCUMENTATION.md"
lines = p.read_text(encoding="utf-8").splitlines()

print(f"DOCUMENTATION.md  lines={len(lines)}  bytes={p.stat().st_size}")

tick = chr(96) * 3
opens = sum(1 for l in lines if l.strip() == tick + "plantuml")
closes = sum(1 for l in lines if l.strip() == tick)
print(f"fences: plantuml opens={opens}, all markers={closes} -> "
      f"{'balanced' if closes % 2 == 0 else 'UNBALANCED'}")

print("\n--- SECTION ORDER ---")
h1 = h2 = 0
for i, l in enumerate(lines, 1):
    if l.startswith("# ") and not l.startswith("##"):
        h1 += 1
        print(f"{i:5d}  {l}")
    elif l.startswith("## "):
        h2 += 1
        print(f"{i:5d}  {l}")
print(f"\nH1 count: {h1} (expected 1)   H2 count: {h2}")

# TOC links must match a heading anchor somewhere in the doc
toc = [l for l in lines if re.match(r"^\s*- \[.+\]\(#", l)]
anchors = set()
for l in lines:
    if l.startswith("#"):
        t = re.sub(r"^#+\s*", "", l).strip()
        a = re.sub(r"[^a-z0-9\- ]", "", t.lower()).replace(" ", "-")
        anchors.add(a)
broken = []
for l in toc:
    a = l.split("(#", 1)[1].rstrip(")")
    if a not in anchors:
        broken.append(a)
print(f"\nTOC entries: {len(toc)}   broken anchors: {len(broken)}")
for b in broken[:12]:
    print("   broken:", b)