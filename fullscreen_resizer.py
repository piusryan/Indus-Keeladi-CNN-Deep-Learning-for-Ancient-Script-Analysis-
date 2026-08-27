#!/usr/bin/env python3
"""
Fullscreen GUI Resizer – no webpage, no resolution issue
- Native Tkinter, true fullscreen, image scales to screen
- Drag red boxes directly, scroll to scale, right-drag to resize
- Saves to .secret_resizer.json (same as web UI) → run_pipeline.py respects it

Run: python fullscreen_resizer.py
Keys: F11 toggle fullscreen, Esc quit, S save, R reset, A add box, Del remove selected
"""
import json
import cv2
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DECODED_DIR = PROJECT_ROOT / "models" / "evaluation_results" / "decoded"
SECRET = PROJECT_ROOT / ".secret_resizer.json"
SRC_DIR = DATA_DIR / "val" / "tamil_brahmi" / "inscriptions_kuviran_atan"

def load_secret():
    if SECRET.exists():
        try:
            return json.loads(SECRET.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

def save_secret(data):
    SECRET.write_text(json.dumps(data, indent=2), encoding="utf-8")

def get_boxes(atan_num):
    # Use same logic as dashboard – get raw tight boxes
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from cnn_potsherd_annotator import PotsherdAnnotator
    annot = PotsherdAnnotator(DATA_DIR, DECODED_DIR)
    src_path = SRC_DIR / f"atan{atan_num}.png"
    import cv2 as _cv2
    potsherd = _cv2.imread(str(src_path))
    if potsherd is None:
        return [], []
    boxes = annot.detect_character_regions(potsherd)
    expected = annot.char_to_indus.get(f"atan{atan_num}", [])
    num_chars = len(expected)
    # Supplement if needed (same as dashboard)
    if len(boxes) < num_chars:
        needed = num_chars - len(boxes)
        boxes = sorted(boxes + annot._distribute_characters(potsherd.shape, num_chars)[:needed], key=lambda b: b[0])
    if len(boxes) > num_chars:
        boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)[:num_chars]
        boxes = sorted(boxes, key=lambda b: b[0])
    # Tighten
    gray = _cv2.cvtColor(potsherd, _cv2.COLOR_BGR2GRAY)
    tight = []
    for (x, y, w, h) in boxes:
        tx, ty, tw, th = annot._tighten_box(gray, x, y, w, h)
        tight.append((tx, ty, tw, th))
    boxes = tight
    # Apply secret
    secret = load_secret()
    boxes = annot._apply_secret_resizer(boxes, atan_num, secret)
    return boxes, expected

class FullscreenResizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fullscreen Resizer – atan")
        self.attributes("-fullscreen", True)
        self.configure(bg="black")
        self.atan_num = tk.IntVar(value=1)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.selected_idx = None
        self.drag_start = None
        self.boxes = []
        self.expected = []
        self.img_path = None
        self.tk_img = None
        self.img_scale = 1.0
        self.img_offset = (0, 0)
        self.orig_img = None

        # Top bar
        top = tk.Frame(self, bg="#1a1a1a")
        top.pack(fill="x", padx=10, pady=5)
        tk.Label(top, text="ATAN:", bg="#1a1a1a", fg="white").pack(side="left")
        ttk.Combobox(top, textvariable=self.atan_num, values=list(range(1, 11)), width=5, state="readonly").pack(side="left", padx=5)
        tk.Button(top, text="Load", command=self.load_atan, bg="#333", fg="white").pack(side="left", padx=5)
        tk.Label(top, text="Scale:", bg="#1a1a1a", fg="white").pack(side="left", padx=(20,5))
        tk.Scale(top, variable=self.scale_var, from_=0.5, to=2.0, resolution=0.05, orient="horizontal", length=150, bg="#1a1a1a", fg="white", highlightthickness=0, command=lambda v: self.update_selected_scale(float(v))).pack(side="left")
        tk.Button(top, text="➕ Add Box", command=self.add_box, bg="#2d5a2d", fg="white").pack(side="left", padx=10)
        tk.Button(top, text="➖ Remove", command=self.remove_selected, bg="#5a2d2d", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="💾 Save", command=self.save, bg="#2d4a7a", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="🔄 Reset", command=self.reset, bg="#4a4a4a", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="▶ Run Pipeline", command=self.run_pipeline, bg="#7a4a2d", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Exit (Esc)", command=self.destroy, bg="#000", fg="white").pack(side="right")

        # Canvas - fullscreen, no scroll
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_wheel(e, delta=120))
        self.canvas.bind("<Button-5>", lambda e: self.on_wheel(e, delta=-120))
        self.bind("<F11>", lambda e: self.attributes("-fullscreen", not self.attributes("-fullscreen")))
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Key-s>", lambda e: self.save())
        self.bind("<Key-a>", lambda e: self.add_box())
        self.bind("<Delete>", lambda e: self.remove_selected())

        self.load_atan()

    def load_atan(self):
        n = int(self.atan_num.get())
        self.img_path = SRC_DIR / f"atan{n}.png"
        if not self.img_path.exists():
            return
        self.orig_img = Image.open(self.img_path).convert("RGB")
        self.boxes, self.expected = get_boxes(n)
        # Apply secret for display
        secret = load_secret()
        # Boxes already include secret via get_boxes, but need to handle extra
        self.selected_idx = 0 if self.boxes else None
        self.draw()

    def draw(self):
        if self.orig_img is None:
            return
        # Fit image to screen fullscreen, keep aspect
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight() - 60
        iw, ih = self.orig_img.size
        scale = min(sw / iw, sh / ih)
        self.img_scale = scale
        nw, nh = int(iw * scale), int(ih * scale)
        ox, oy = (sw - nw)//2, (sh - nh)//2 + 30
        self.img_offset = (ox, oy)
        img_resized = self.orig_img.resize((nw, nh), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.tk_img)
        # Draw red boxes scaled
        for i, (x, y, w, h) in enumerate(self.boxes):
            # Scale to screen
            sx, sy = int(x * scale) + ox, int(y * scale) + oy
            sw_, sh_ = int(w * scale), int(h * scale)
            color = "#00ff00" if i == self.selected_idx else "red"
            width = 3 if i == self.selected_idx else 2
            self.canvas.create_rectangle(sx, sy, sx+sw_, sy+sh_, outline=color, width=width, tags=f"box{i}")
            # Label
            brahmi = self.expected[i][0] if i < len(self.expected) else f"extra{i}"
            self.canvas.create_text(sx+2, sy-10, text=f"B:{brahmi}", fill="cyan", anchor="nw", font=("Arial", 8), tags=f"label{i}")
            # Size hint
            self.canvas.create_text(sx+2, sy+sh_+2, text=f"{w}x{h}", fill="lightgreen", anchor="nw", font=("Arial", 7), tags=f"size{i}")

    def on_click(self, event):
        # Find closest box
        ox, oy = self.img_offset
        x = (event.x - ox) / self.img_scale
        y = (event.y - oy) / self.img_scale
        best = None
        best_dist = 1e9
        for i, (bx, by, bw, bh) in enumerate(self.boxes):
            cx, cy = bx + bw/2, by + bh/2
            dist = abs(x - cx) + abs(y - cy)
            if dist < best_dist and bx <= x <= bx+bw and by <= y <= by+bh:
                best_dist = dist
                best = i
        if best is not None:
            self.selected_idx = best
            self.drag_start = (event.x, event.y, self.boxes[best])
            self.draw()

    def on_drag(self, event):
        if self.selected_idx is None or self.drag_start is None:
            return
        dx = (event.x - self.drag_start[0]) / self.img_scale
        dy = (event.y - self.drag_start[1]) / self.img_scale
        i = self.selected_idx
        x, y, w, h = self.drag_start[2]
        self.boxes[i] = (int(x + dx), int(y + dy), w, h)
        self.draw()

    def on_wheel(self, event, delta=None):
        if self.selected_idx is None:
            return
        d = delta if delta is not None else event.delta
        scale = 1.1 if d > 0 else 0.9
        i = self.selected_idx
        x, y, w, h = self.boxes[i]
        nw, nh = int(w * scale), int(h * scale)
        nx, ny = x + (w - nw)//2, y + (h - nh)//2
        self.boxes[i] = (nx, ny, nw, nh)
        self.draw()

    def on_release(self, event):
        self.drag_start = None

    def update_selected_scale(self, val):
        if self.selected_idx is not None:
            i = self.selected_idx
            x, y, w, h = self.boxes[i]
            # Scale around center
            nw, nh = int(w * (val / 1.0)), int(h * (val / 1.0))
            # For live, we need original w,h – store separately
            pass

    def add_box(self):
        # Add centered box
        if self.orig_img:
            w, h = self.orig_img.size
            self.boxes.append((w//2 - 37, h//2 - 55, 75, 110))
            if len(self.expected) <= len(self.boxes)-1:
                self.expected.append(("ma", "P145", 0.80))
            self.selected_idx = len(self.boxes) - 1
            self.draw()

    def remove_selected(self):
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.boxes):
            del self.boxes[self.selected_idx]
            if self.selected_idx < len(self.expected):
                del self.expected[self.selected_idx]
            self.selected_idx = max(0, min(self.selected_idx, len(self.boxes)-1)) if self.boxes else None
            self.draw()

    def save(self):
        # Save to .secret_resizer.json by diffing against original detected boxes
        n = int(self.atan_num.get())
        # Get original tight boxes without secret
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from cnn_potsherd_annotator import PotsherdAnnotator
        annot = PotsherdAnnotator(DATA_DIR, DECODED_DIR)
        import cv2 as _cv2
        potsherd = _cv2.imread(str(SRC_DIR / f"atan{n}.png"))
        orig_boxes = annot.detect_character_regions(potsherd)
        # Simplified: just save current boxes as secret delta from orig
        # For now, save as absolute boxes in secret
        secret = load_secret()
        key = f"atan{n}"
        secret[key] = {}
        # Need orig tight boxes to compute delta
        gray = _cv2.cvtColor(potsherd, _cv2.COLOR_BGR2GRAY)
        tight = []
        for (x, y, w, h) in orig_boxes[:len(self.boxes)]:
            # If we have fewer orig than current, use current as is
            pass
        # Instead, save current boxes directly as absolute overrides
        for i, (x, y, w, h) in enumerate(self.boxes):
            # Find original for this index if exists
            if i < len(orig_boxes):
                ox, oy, ow, oh = orig_boxes[i]
                # Compute scale and dx/dy relative to orig
                # Use tight orig
                tx, ty, tw, th = annot._tighten_box(gray, ox, oy, ow, oh) if i < len(orig_boxes) else (ox, oy, ow, oh)
                scale_w = w / tw if tw else 1
                scale_h = h / th if th else 1
                scale = (scale_w + scale_h)/2
                dx = (x + w//2) - (tx + tw//2)
                dy = (y + h//2) - (ty + th//2)
                secret[key][str(i)] = {"scale": round(scale, 2), "dx": int(dx), "dy": int(dy), "w": int(w), "h": int(h)}
            else:
                # Extra box
                secret[key][str(i)] = {"scale": 1.0, "dx": 0, "dy": 0, "w": int(w), "h": int(h), "brahmi": self.expected[i][0] if i < len(self.expected) else "ma", "indus": self.expected[i][1] if i < len(self.expected) else "P145"}
        save_secret(secret)
        # Also need to handle hidden boxes (removed) – already removed
        # Re-annotate to apply
        annot.annotate_potsherd(n)
        self.load_atan()
        print(f"Saved {key} -> {secret[key]}")

    def reset(self):
        secret = load_secret()
        key = f"atan{int(self.atan_num.get())}"
        if key in secret:
            del secret[key]
            save_secret(secret)
        self.load_atan()

    def run_pipeline(self):
        import subprocess, sys
        self.title("Running pipeline...")
        result = subprocess.run([sys.executable, "run_pipeline.py"], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT))
        print(result.stdout[-2000:])
        self.title("Fullscreen Resizer – done")

if __name__ == "__main__":
    app = FullscreenResizer()
    app.mainloop()
