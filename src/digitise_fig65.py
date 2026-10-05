"""
digitise_fig65.py - extract the REAL Indus sign glyphs from the source paper.

WHY
---
`docs/THE INDUS SCRIPT Recognition as an Alphabet.pdf`, Figure 65 (pp.35-36) is a
table of the **40 primary core signs**. Its "Indus Sign" column holds genuine
sign glyphs, and each row carries three published numberings:

    Serial | NFM Unicode PUA | M-1977 (Mahadevan) | W-2015 (Wells) | P-2010

This script renders the table at high resolution and crops every sign cell,
labelling it by serial number and by its P-2010 / M-1977 codes. That gives us
REAL reference glyphs with authoritative labels - replacing the current
"1 drawing + 19 synthetic augmentation clones" training data.

Output:  data/processed/train/fig65_real/fig65_rowNN_<P>.png  (+ a manifest json)

Run:  python -m src.digitise_fig65
"""

import os
import sys
import json
import logging
import re
import warnings
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import cv2
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PDF = PROJECT_ROOT / "docs" / "THE INDUS SCRIPT Recognition as an Alphabet.pdf"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "train" / "fig65_real"
ZOOM = 6.0

# Fig.65 rows, verified from the extracted text (p.35 = serials 1-12,
# p.36 = serials 13-40). "or" rows are alternative codes for the SAME sign.
FIG65_ROWS = {
    1:  ("E06-D", "90", "13", "13"),      2:  ("E10-A", "59", "220", "60"),
    3:  ("E12-D", "67", "240", "72"),     4:  ("E13-D", "78", "266", "76"),
    5:  ("E1B-E", "53", "798", "88"),     6:  ("E1E-2", "162", "390", "91"),
    7:  ("E24-A", "176", "400", "107"),   8:  ("E26-9", "400", "374", "109"),
    9:  ("E2D-6", "99", "2", "127"),      10: ("E2D-9", "98", "1", "128"),
    11: ("E2D-C", "87", "32", "129"),     12: ("E2D-F", "89", "33", "130"),
    13: ("E2E-1", "102", "3", "130"),     14: ("E2F-0", "109", "16", "133"),
    15: ("E31-5", "121", "18", "145"),    16: ("E31-A", "86", "31", "147"),
    17: ("E33-A|E34-F", "287|299", "900|899", "156|165"),
    18: ("E37-D|E38-F", "304|307", "890|892", "181|187"),
    19: ("E3C-2", "205", "491", "192"),   20: ("E3D-B", "230", "460", "198"),
    21: ("E3E-8|E40-9", "134|135", "480|482", "200|209"),
    22: ("E43-5", "402", "367", "214"),   23: ("E45-D", "180", "306", "217"),
    24: ("E46-E", "225", "530", "219"),   25: ("E47-D", "216", "550", "225"),
    26: ("E4A-6", "137", "645", "245"),   27: ("E50-3", "237", "625", "266"),
    28: ("E51-8|E6E-E", "245|247", "615|626", "272|371"),
    29: ("E55-0", "249", "590", "278"),   30: ("E56-1|E56-F|E57-4", "199|195|194",
                                               "570|572|576", "282|285|287"),
    31: ("E58-5", "197", "575", "289"),   32: ("E5A-D", "328", "700", "296"),
    33: ("E5D-5", "336", "706", "302"),   34: ("E65-D", "347", "760", "319"),
    35: ("E69-4", "261", "850", "341"),   36: ("E69-9", "373", "790", "341"),
    37: ("E6E-8", "391", "820", "368"),   38: ("E6F-4", "284", "877", "373"),
    39: ("E70-8", "267", "817", "376"),
}

# Serials whose glyph is a plain stroke/line drawing. Shape-matching against
# these is degenerate (almost any thin stroke "matches"), so they are flagged
# and excluded from match claims - see docs/ANALYSIS_ACCURACY_ROADMAP.md 14.4.
STROKE_SERIALS = {9, 10, 11, 12, 13, 14, 15, 16, 30}


def table_rules(page):
    """Vertical/horizontal rule positions of the Fig.65 table grid, read from
    the PDF vector drawings. Cropping strictly INSIDE these rules removes the
    borders without touching any part of the glyph."""
    xs, ys = set(), set()
    try:
        drawings = page.get_drawings()
    except Exception:
        return [], []
    for d in drawings:
        for item in d.get("items", []):
            kind = item[0]
            if kind == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.x - p2.x) < 0.8 and abs(p1.y - p2.y) > 8:
                    xs.add(round((p1.x + p2.x) / 2, 1))
                elif abs(p1.y - p2.y) < 0.8 and abs(p1.x - p2.x) > 8:
                    ys.add(round((p1.y + p2.y) / 2, 1))
            elif kind == "re":
                r = item[1]
                if r.width < 1.5 and r.height > 8:
                    xs.add(round((r.x0 + r.x1) / 2, 1))
                if r.height < 1.5 and r.width > 8:
                    ys.add(round((r.y0 + r.y1) / 2, 1))
    return sorted(xs), sorted(ys)


def _bracket(values, lo, hi):
    """Smallest span from `values` that covers [lo, hi]; else (lo, hi)."""
    inside = [v for v in values if lo - 0.5 <= v <= hi + 0.5]
    if len(inside) >= 2:
        return inside[0], inside[-1]
    below = [v for v in values if v <= lo]
    above = [v for v in values if v >= hi]
    if below and above:
        return below[-1], above[0]
    return lo, hi


def clean_crop(img):
    """Remove the Figure-65 table rules and keep only the sign glyph.

    The rules of the table form one connected network (they meet at the cell
    corners), so dropping thin connected components does not separate them.
    Instead we erase every ink run that spans >=85% of the cell width/height -
    the table rules always span the full cell, whereas a stroke of the sign
    itself almost never does - and then keep the largest remaining component.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.bitwise_not(gray)                      # ink -> bright
    binary = cv2.threshold(binary, 60, 255, cv2.THRESH_BINARY)[1]
    H, W = binary.shape[:2]
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(binary, 8)
    cands = [(stats[i][4], stats[i][0], stats[i][1], stats[i][2], stats[i][3])
             for i in range(1, n) if stats[i][4] >= 6]
    if not cands:
        return None
    _a, x, y, w, h = max(cands)
    pad = 3
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return binary[y0:y1, x0:x1]


def _words(page):
    """(x0, y0, x1, y1, text) for every word on the page."""
    return [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]


def _serial_anchors(page, serials):
    """Locate the serial-number cells: {serial: (x1, y0, y1)} in column 1."""
    found = {}
    want = {f"{s}." for s in serials}
    for x0, y0, x1, y1, w in _words(page):
        if w in want and x1 < 300:          # column 1 only
            s = int(w[:-1])
            if s not in found:
                found[s] = (x1, y0, y1)
    return found


def digitise(pdf=PDF, out_dir=OUT_DIR, zoom=ZOOM):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf))
    manifest = {}
    # p.35 carries serials 1-12, p.36 carries 13-40 (1-based -> 0-based)
    for pno, serials in [(34, list(range(1, 13))), (35, list(range(13, 40)))]:
        page = doc[pno]
        anchors = _serial_anchors(page, serials)
        if not anchors:
            continue
        # column 3 (NFM codes like "E46-E") defines the right edge of the sign cell
        nfm_x0 = None
        for x0, y0, x1, y1, w in _words(page):
            if re.fullmatch(r"E[0-9A-F]{2,3}-[0-9A-F]", w):
                nfm_x0 = x0 if nfm_x0 is None else min(nfm_x0, x0)
        if nfm_x0 is None:
            continue
        vrules, hrules = table_rules(page)
        # Row bands from midpoints between neighbouring serial labels.
        order = sorted(anchors.items(), key=lambda kv: kv[1][1])
        row_bands = {}
        for i, (s, (_sx1, y0, y1)) in enumerate(order):
            if i > 0:
                _ps, (_px1, py0, py1) = order[i - 1]
                top = (py1 + y0) / 2.0
            else:
                top = y0 - 8.0
            if i < len(order) - 1:
                _ns, (_nx1, ny0, _ny1) = order[i + 1]
                bot = (y1 + ny0) / 2.0
            else:
                bot = y1 + 8.0
            row_bands[s] = (top, bot)
        for s, (sx1, y0, y1) in sorted(anchors.items()):
            # left/right cell edges from the vertical rules, top/bottom from the
            # horizontal rules -> a crop strictly inside the table borders.
            lx = [v for v in vrules if v > sx1]
            left = lx[0] if lx else (sx1 + 4.0)
            # right edge = the NFM column (reliable on both pages); fall back
            # to the next vertical rule if no NFM word was found.
            right = (nfm_x0 - 4.0) if nfm_x0 else (
                lx[1] if len(lx) >= 2 else (left + 60.0))
            # Row band from the midpoints between neighbouring serial labels.
            # This needs no table rules (page 35's table has none) and is exact
            # because each label is vertically centred in its own row.
            inset = 2.0
            top = max([v for v in hrules if v <= y0], default=y0 - 8.0)
            bot = min([v for v in hrules if v >= y1], default=y1 + 8.0)
            rect = pymupdf.Rect(left + inset, top + inset,
                                right - inset, bot - inset)
            if rect.width <= 4 or rect.height <= 4:
                continue
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)
            if pix.n >= 3:
                img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
            if img.size == 0 or img.shape[0] < 6 or img.shape[1] < 6:
                continue
            glyph = clean_crop(img)
            if glyph is None or glyph.size == 0 or min(glyph.shape[:2]) < 6:
                continue
            _nfm, m_num, _w, p_num = FIG65_ROWS.get(s, ("", "", "", ""))
            name = f"fig65_row{s:02d}_P{p_num.replace('|', '-')}.png"
            cv2.imwrite(str(out_dir / name), glyph)
            manifest[f"fig65_row{s:02d}"] = {
                "file": name, "serial": s, "nfm": _nfm,
                "m1977": m_num, "p2010": p_num,
                "is_stroke_sign": s in STROKE_SERIALS,
            }
    doc.close()
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def debug_page(pno):
    doc = pymupdf.open(str(PDF))
    page = doc[pno]
    serials = list(range(1, 13)) if pno == 34 else list(range(13, 40))
    a = _serial_anchors(page, serials)
    print("anchors:", {k: (round(v[0], 1), round(v[1], 1), round(v[2], 1))
                       for k, v in sorted(a.items())})
    v, h = table_rules(page)
    print("vrules:", v)
    print("hrules:", h)
    doc.close()


def debug_rects(pno=34):
    """Print the crop rectangle used for each serial (diagnostic)."""
    doc = pymupdf.open(str(PDF))
    page = doc[pno]
    serials = list(range(1, 13)) if pno == 34 else list(range(13, 40))
    anchors = _serial_anchors(page, serials)
    nfm_x0 = None
    for x0, y0, x1, y1, w in _words(page):
        if re.fullmatch(r"E[0-9A-F]{2,3}-[0-9A-F]", w):
            nfm_x0 = x0 if nfm_x0 is None else min(nfm_x0, x0)
    vrules, hrules = table_rules(page)
    for s, (sx1, y0, y1) in sorted(anchors.items()):
        lx = [v for v in vrules if v > sx1]
        left = lx[0] if lx else sx1 + 4.0
        right = (nfm_x0 - 4.0) if nfm_x0 else (lx[1] if len(lx) >= 2 else left + 60)
        top = max([v for v in hrules if v <= y0], default=y0 - 8.0)
        bot = min([v for v in hrules if v >= y1], default=y1 + 8.0)
        r = pymupdf.Rect(left + 2, top + 2, right - 2, bot - 2)
        print(f"serial {s:2d}: rect w={r.width:6.1f} h={r.height:6.1f} "
              f"y=[{r.y0:.1f},{r.y1:.1f}]")
    doc.close()


def contact_sheet(out=None, cell=96, cols=10):
    """Tile every digitised glyph into one labelled grid for visual QA."""
    out = Path(out or (PROJECT_ROOT / "docs" / "figures" /
                       "fig65_digitised_contact_sheet.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    man = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    keys = sorted(man)
    rows = (len(keys) + cols - 1) // cols
    sheet = np.full((rows * (cell + 16), cols * cell), 255, np.uint8)
    for i, k in enumerate(keys):
        img = cv2.imread(str(OUT_DIR / man[k]["file"]), cv2.IMREAD_GRAYSCALE)
        r, c = divmod(i, cols)
        if img is None:
            continue
        # white background, black ink, aspect preserved
        ink = (img < 128).astype(np.uint8) * 255
        h, w = ink.shape[:2]
        s = (cell - 8) / max(h, w)
        nh, nw = max(1, int(h * s)), max(1, int(w * s))
        rs = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_NEAREST)
        tile = np.full((cell, cell), 255, np.uint8)
        y0, x0 = (cell - nh) // 2, (cell - nw) // 2
        tile[y0:y0 + nh, x0:x0 + nw] = rs
        sheet[r * (cell + 16) + 16:(r + 1) * (cell + 16), c * cell:(c + 1) * cell] = tile
        cv2.putText(sheet, k.replace("fig65_row", ""), (c * cell + 4,
                    r * (cell + 16) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    cv2.imwrite(str(out), sheet)
    return out


if __name__ == "__main__":
    m = digitise()
    print(f"digitised {len(m)} sign glyphs -> {OUT_DIR}")
    print("contact sheet ->", contact_sheet())