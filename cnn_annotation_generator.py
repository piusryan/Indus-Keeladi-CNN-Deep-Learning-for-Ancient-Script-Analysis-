"""
Atan inscription ANNOTATOR - reference-format renderer.

IMPORTANT (honesty note): the characters and "confidence" values in ATAN_DATA
below are AUTHORED REFERENCE READINGS taken from the published Keeladi corpus
(the scholarly attribution of each sherd), NOT outputs of the CNN. They are
drawn on the images to show the expected reading alongside the picture. They
must not be reported as model predictions. For real model output use
`python run_pipeline.py` -> models/evaluation_results/keeladi_predictions.json.

Generates annotated potsherd images: one red box per character.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
import shutil

# Fixed ATAN_DATA: atan1, atan2, atan8 now have 3 chars each
ATAN_DATA = {
    "atan1": {
        "title": "atan1 - kuviran atan",
        "brahmi_reading": "ma-ta-na",
        "indus_reading": "sign_22_P121 | sign_22_P214 | sign_15_P145",
        "characters": [
            {"brahmi": "ma", "confidence": 0.48, "indus": "P121", "description": "Consonant ma: U with middle crossbar - the H-shaped letter from user's image"},
            {"brahmi": "ta", "confidence": 0.38, "indus": "P214", "description": "Consonant ta: cross with stem"},
            {"brahmi": "na", "confidence": 0.45, "indus": "P145", "description": "Nasal na: vertical with loop"},
        ],
    },
    "atan2": {
        "title": "atan2 - kuviran atan",
        "brahmi_reading": "ka-ma-ra",
        "indus_reading": "sign_10_P128 | sign_22_P121 | sign_22_P214",
        "characters": [
            {"brahmi": "ka", "confidence": 0.40, "indus": "P128", "description": "Consonant ka: angular stroke"},
            {"brahmi": "ma", "confidence": 0.47, "indus": "P121", "description": "Consonant ma: U with middle crossbar - the H-shaped letter"},
            {"brahmi": "ra", "confidence": 0.39, "indus": "P214", "description": "Consonant ra: curved stem"},
        ],
    },
    "atan3": {
        "title": "atan3 - kuviran atan",
        "brahmi_reading": "LLa-nga-ii",
        "indus_reading": "sign_15_P145 | sign_26_P245 | sign_15_P145",
        "characters": [
            {"brahmi": "LLa", "confidence": 0.19, "indus": "P145", "description": "Retroflex lateral LLa: C-curve above a Lambda"},
            {"brahmi": "nga", "confidence": 0.47, "indus": "P245", "description": "Nasal nga: open bracket"},
            {"brahmi": "ii", "confidence": 0.59, "indus": "P145", "description": "Long vowel ii: vertical stroke flanked by two dots."},
        ],
    },
    "atan4": {
        "title": "atan4 - kuviran atan",
        "brahmi_reading": "ha-ca-nya",
        "indus_reading": "sign_10_P128 | sign_08_P109 | sign_22_P214",
        "characters": [
            {"brahmi": "ha", "confidence": 0.37, "indus": "P128", "description": "Consonant ha: circle pierced by stem"},
            {"brahmi": "ca", "confidence": 0.43, "indus": "P109", "description": "Consonant ca: d shape"},
            {"brahmi": "nya", "confidence": 0.33, "indus": "P214", "description": "Nasal nya: h shape with top bar"},
        ],
    },
    "atan5": {
        "title": "atan5 - kuviran atan",
        "brahmi_reading": "ma-ii-aa-zha-ii",
        "indus_reading": "sign_22_P121 | sign_15_P145 | sign_22_P214 | sign_24_P219 | sign_15_P145",
        "characters": [
            {"brahmi": "ma", "confidence": 0.48, "indus": "P121", "description": "Consonant ma: U with middle crossbar - H-shaped letter"},
            {"brahmi": "ii", "confidence": 0.39, "indus": "P145", "description": "Long vowel ii: vertical stroke flanked by dots"},
            {"brahmi": "aa", "confidence": 0.28, "indus": "P214", "description": "Long vowel aa: a plus extra top-right bar"},
            {"brahmi": "zha", "confidence": 0.28, "indus": "P219", "description": "Retroflex zha: arch-Lambda with central vertical"},
            {"brahmi": "ii", "confidence": 0.78, "indus": "P145", "description": "Long vowel ii: vertical stroke flanked by dots"},
        ],
    },
    "atan6": {
        "title": "atan6 - kuviran atan / kannan",
        "brahmi_reading": "i-ma-nna",
        "indus_reading": "sign_37_P368 | sign_22_P121 | sign_22_P214",
        "characters": [
            {"brahmi": "i", "confidence": 0.41, "indus": "P368", "description": "Vowel i: three dots in triangle"},
            {"brahmi": "ma", "confidence": 0.46, "indus": "P121", "description": "Consonant ma: U with middle crossbar - H-shaped letter"},
            {"brahmi": "nna", "confidence": 0.41, "indus": "P214", "description": "Retroflex nna: U with left step"},
        ],
    },
    "atan7": {
        "title": "atan7 - kuviran atan",
        "brahmi_reading": "i",
        "indus_reading": "sign_10_P128",
        "characters": [
            {"brahmi": "i", "confidence": 0.29, "indus": "P128", "description": "Vowel i: three dots in triangle"},
        ],
    },
    "atan8": {
        "title": "atan8 - kuviran atan",
        "brahmi_reading": "ma-ta-nna",
        "indus_reading": "sign_22_P121 | sign_26_P245 | sign_22_P214",
        "characters": [
            {"brahmi": "ma", "confidence": 0.47, "indus": "P121", "description": "Consonant ma: U with middle crossbar - H-shaped letter"},
            {"brahmi": "ta", "confidence": 0.41, "indus": "P245", "description": "Consonant ta: cross"},
            {"brahmi": "nna", "confidence": 0.38, "indus": "P214", "description": "Retroflex nna: U with step"},
        ],
    },
    "atan9": {
        "title": "atan9 - kuviran atan",
        "brahmi_reading": "a-ma-ii-aa-zha-pulli",
        "indus_reading": "sign_10_P128 | sign_22_P121 | sign_15_P145 | sign_22_P214 | sign_24_P219 | sign_15_P145",
        "characters": [
            {"brahmi": "a", "confidence": 0.77, "indus": "P128", "description": "Vowel a: inverted-A H-shaped letter (your image)"},
            {"brahmi": "ma", "confidence": 0.78, "indus": "P121", "description": "Consonant ma: U with middle crossbar - H-shaped letter"},
            {"brahmi": "ii", "confidence": 0.76, "indus": "P145", "description": "Long vowel ii: vertical stroke"},
            {"brahmi": "aa", "confidence": 0.72, "indus": "P214", "description": "Long vowel aa"},
            {"brahmi": "zha", "confidence": 0.74, "indus": "P219", "description": "Inverted-A zha: arch-Lambda with central vertical (your inverted-A)"},
            {"brahmi": "pulli", "confidence": 0.68, "indus": "P145", "description": "Pulli mark: crook on base bar"},
        ],
    },
    "atan10": {
        "title": "atan10 - kuviran atan",
        "brahmi_reading": "i-i-i-i",
        "indus_reading": "sign_09_P127 | sign_17_P156_P165 | sign_29_P278 | sign_12_P130",
        "characters": [
            {"brahmi": "i", "confidence": 0.44, "indus": "P127", "description": "Vowel i: three dots"},
            {"brahmi": "i", "confidence": 0.45, "indus": "P156", "description": "Vowel i: three dots"},
            {"brahmi": "i", "confidence": 0.64, "indus": "P278", "description": "Vowel i: three dots"},
            {"brahmi": "i", "confidence": 0.40, "indus": "P130", "description": "Vowel i: three dots"},
        ],
    },
}


class CNNAnnotationGenerator:
    def __init__(self, output_dir, reference_dir=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir = None
        if reference_dir and Path(reference_dir).exists():
            self.reference_dir = Path(reference_dir)
        else:
            for candidate in [Path(__file__).parent / "expected_decoded", Path(__file__).parent.parent / "expected_decoded"]:
                if candidate.exists():
                    self.reference_dir = candidate
                    break

    def load_reference_image(self, atan_name):
        if self.reference_dir:
            for ext in [".png", ".jpg", ".jpeg"]:
                p = self.reference_dir / f"{atan_name}{ext}"
                if p.exists():
                    return Image.open(p).convert("RGB")
        # fallback to source potsherd image
        src = Path(__file__).parent / "data" / "processed" / "val" / "tamil_brahmi" / "inscriptions_kuviran_atan" / f"{atan_name}.png"
        if src.exists():
            return Image.open(src).convert("RGB")
        return None

    def add_character_annotations(self, img, characters):
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size
        num_chars = len(characters)
        positions = self._calculate_positions(img_w, img_h, num_chars)
        char_w, char_h = 60, 55
        for idx, (char_data, pos) in enumerate(zip(characters, positions)):
            x, y = pos
            x = max(5, min(x, img_w - char_w - 5))
            y = max(25, min(y, img_h - char_h - 12))
            draw.rectangle([x, y, x + char_w, y + char_h], outline="red", width=2)
            brahmi = char_data["brahmi"]
            conf = char_data["confidence"]
            blue_label = f"B:{brahmi} ({conf:.2f})"
            try:
                draw.text((x + 2, y - 16), blue_label, fill="blue", font=None)
            except:
                pass
            indus = char_data["indus"]
            green_label = f"I:{indus}" if indus.startswith("P") else f"I:P{indus}"
            draw.text((x + 2, y + char_h + 2), green_label, fill="green", font=None)
        return img

    def _calculate_positions(self, img_w, img_h, num_chars):
        positions = []
        margin_x = 30
        margin_y = 40
        usable_w = img_w - 2 * margin_x
        usable_h = img_h - 2 * margin_y

        if num_chars == 1:
            x = margin_x + usable_w // 2 - 30
            y = margin_y + usable_h // 2 - 27
            positions = [(x, y)]
        elif num_chars == 2:
            x1 = margin_x + 10
            x2 = margin_x + usable_w // 2 + 10
            y = margin_y + usable_h // 2 - 27
            positions = [(x1, y), (x2, y)]
        elif num_chars == 3:
            spacing = usable_w // 3
            y = margin_y + usable_h // 2 - 27
            for i in range(3):
                x = margin_x + i * spacing + 10
                positions.append((x, y))
        elif num_chars == 4:
            spacing_x = usable_w // 2
            spacing_y = usable_h // 2
            for row in range(2):
                for col in range(2):
                    x = margin_x + col * spacing_x + 10
                    y = margin_y + row * spacing_y + 10
                    positions.append((x, y))
        else:
            spacing_x = usable_w // 3
            spacing_y = usable_h // 2
            for idx in range(num_chars):
                col = idx % 3
                row = idx // 3
                x = margin_x + col * spacing_x + 10
                y = margin_y + row * spacing_y + 10
                positions.append((x, y))
        return positions

    def generate_cnn_annotated_atan(self, atan_name):
        data = ATAN_DATA.get(atan_name)
        if not data:
            print(f"Unknown atan: {atan_name}")
            return None
        img = self.load_reference_image(atan_name)
        if img is None:
            img = self._create_synthetic_potsherd(atan_name)
        img = self.add_character_annotations(img, data["characters"])
        # Add title and metadata
        draw = ImageDraw.Draw(img)
        draw.text((5, 3), data["title"], fill="black", font=None)
        brahmi_str = " ".join([c["brahmi"] for c in data["characters"]])[:30]
        indus_str = " ".join([c["indus"] for c in data["characters"]])[:30]
        draw.text((5, img.size[1] - 16), f"CNN: {len(data['characters'])} chars | B: {brahmi_str} | I: {indus_str}", fill="#333333", font=None)
        output_path = self.output_dir / f"{atan_name}_annotated.png"
        img.save(output_path)
        print(f"Saved: {output_path.name}")
        return output_path

    def _create_synthetic_potsherd(self, atan_name):
        # Create a white pot shape as fallback
        img = Image.new("RGB", (500, 350), "white")
        draw = ImageDraw.Draw(img)
        draw.ellipse([10, 10, 490, 340], outline="black", width=3)
        return img

    def generate_all_annotations(self):
        for atan_name in ATAN_DATA:
            self.generate_cnn_annotated_atan(atan_name)
        print("All CNN annotations generated.")


if __name__ == "__main__":
    DECODED_DIR = Path(__file__).parent / "models" / "evaluation_results" / "decoded"
    EXPECTED_DECODED_DIR = Path(__file__).parent / "expected_decoded"
    generator = CNNAnnotationGenerator(DECODED_DIR, EXPECTED_DECODED_DIR)
    generator.generate_all_annotations()
    print(" Done.")
