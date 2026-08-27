"""
Evaluation Script for Indus-Keeladi CNN Project
Matches Keeladi graffiti signs against Indus alphabet to find civilization links
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.image_normalization import ImageNormalizer
from src.models.indus_classifier_cnn import IndusClassifierCNN


# Expected archaeological correspondences: match folder -> training class name.
# Fill these in from the research notebook sources.  `None` means the expected
# Indus sign has no training class yet, so top-1 correctness cannot be scored
# for that folder (it is reported as "unmapped" instead).
EXPECTED_MATCH_MAP = {
    "match_Indus_225": "sign_25_P225_Cross",
    "match_Indus_307": "sign_41_P307",
    "match_Indus_318": "sign_42_P318",
    "match_Indus_365": "sign_43_P365",
}

# Training classes whose name starts with this prefix are "not an Indus sign"
# rejection classes (general graffiti / background), not real signs.
REJECTION_PREFIX = "zz_"


def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class KeeladiEvaluator:
    """
    Evaluates Keeladi graffiti against trained Indus script model
    """
    
    def __init__(self, model_path, data_dir):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            data_dir: Root directory for data
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        
        self.normalizer = ImageNormalizer(target_size=(64, 64))
        self.classifier = IndusClassifierCNN()
        self.classifier.load_model(str(self.model_path))
        
        # Load class names
        class_names_path = self.model_path.parent / f"{self.model_path.stem}_classes.txt"
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        
        self.logger.info(f"Loaded model with {len(self.class_names)} classes")

    def _rejection_class_index(self):
        """Index of the 'not an Indus sign' rejection class, or None."""
        for idx, name in enumerate(self.class_names):
            if name.startswith(REJECTION_PREFIX):
                return idx
        return None

    def _is_negative_folder(self, folder_name):
        """Folders that must NOT be matched to an Indus sign (controls)."""
        return (folder_name == "general_keeladi_graffiti"
                or folder_name.startswith("tamil_brahmi_"))
    
    def _list_image_files(self, directory):
        """List all common image format files in a directory"""
        directory = Path(directory)
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        return [f for f in directory.iterdir() if f.suffix.lower() in extensions]
    
    def load_keeladi_validation_set(self):
        """
        Load Keeladi validation images including all subfolder categories
        
        Returns:
            Dictionary of match folders with their images and file paths
        """
        self.logger.info("Loading Keeladi validation datasets...")
        val_dir = self.data_dir / "processed" / "val" / "keeladi"
        tb_dir = self.data_dir / "processed" / "val" / "tamil_brahmi"
        
        validation_data = {}
        validation_files = {}
        
        # Load direct match folders
        match_folders = [
            "match_Indus_225",
            "match_Indus_307", 
            "match_Indus_365",
            "match_Indus_318"
        ]
        
        for match_folder in match_folders:
            match_dir = val_dir / match_folder
            if match_dir.exists():
                images = []
                files = []
                image_files = self._list_image_files(match_dir)
                
                for image_file in image_files:
                    try:
                        processed = self.normalizer.process_image(image_file)
                        images.append(processed)
                        files.append(image_file.name)
                    except Exception as e:
                        self.logger.error(f"Error loading {image_file}: {e}")
                
                if images:
                    validation_data[match_folder] = np.array(images)
                    validation_files[match_folder] = files
                    self.logger.info(f"  {match_folder}: {len(images)} image(s)")
        
        # Load general Keeladi graffiti
        general_dir = val_dir / "general_keeladi_graffiti"
        if general_dir.exists():
            images = []
            files = []
            image_files = self._list_image_files(general_dir)
            
            for image_file in image_files:
                try:
                    processed = self.normalizer.process_image(image_file)
                    images.append(processed)
                    files.append(image_file.name)
                except Exception as e:
                    self.logger.error(f"Error loading {image_file}: {e}")
            
            if images:
                validation_data["general_keeladi_graffiti"] = np.array(images)
                validation_files["general_keeladi_graffiti"] = files
                self.logger.info(f"  general_keeladi_graffiti: {len(images)} image(s)")
        
        # Load Tamil-Brahmi inscriptions if available
        tamil_brahmi_dir = tb_dir
        if tamil_brahmi_dir.exists():
            for subfolder in tamil_brahmi_dir.iterdir():
                if subfolder.is_dir():
                    images = []
                    files = []
                    image_files = self._list_image_files(subfolder)
                    
                    for image_file in image_files:
                        try:
                            processed = self.normalizer.process_image(image_file)
                            images.append(processed)
                            files.append(image_file.name)
                        except Exception as e:
                            self.logger.error(f"Error loading {image_file}: {e}")
                    
                    if images:
                        key = f"tamil_brahmi_{subfolder.name}"
                        validation_data[key] = np.array(images)
                        validation_files[key] = files
                        self.logger.info(f"  {key}: {len(images)} image(s)")
        
        self.validation_files = validation_files
        return validation_data
    
    def predict_keeladi_matches(self, validation_data, threshold=0.5):
        """
        Predict Indus sign matches for Keeladi graffiti
        Uses a lower threshold by default for research discovery
        
        Args:
            validation_data: Dictionary of validation images
            threshold: Confidence threshold for positive match
            
        Returns:
            Dictionary of predictions for each validation set
        """
        self.logger.info(f"Predicting Indus sign matches (threshold={threshold})...")
        predictions = {}
        rejection_idx = self._rejection_class_index()
        
        for folder_name, images in validation_data.items():
            # Reshape for CNN
            X = images.reshape(images.shape[0], 64, 64, 1)
            
            # Get predictions
            pred_probs = self.classifier.predict(X)
            pred_classes = np.argmax(pred_probs, axis=1)
            pred_confidences = np.max(pred_probs, axis=1)
            
            # A prediction of the rejection class means "not an Indus sign"
            if rejection_idx is not None:
                is_rejection = pred_classes == rejection_idx
            else:
                is_rejection = np.zeros(len(pred_classes), dtype=bool)
            
            # Get top-3 predictions per image for research analysis
            top3_classes = np.argsort(pred_probs, axis=1)[:, -3:][:, ::-1]
            top3_probs = np.sort(pred_probs, axis=1)[:, -3:][:, ::-1]
            
            # Filter by threshold; predictions of the rejection class never
            # count as Indus matches no matter how confident they are
            high_confidence_mask = (pred_confidences >= threshold) & ~is_rejection
            high_confidence_matches = pred_classes[high_confidence_mask]
            high_confidence_scores = pred_confidences[high_confidence_mask]
            
            folder_predictions = {
                'all_predictions': pred_classes,
                'all_confidences': pred_confidences,
                'is_rejection': is_rejection,
                'top3_classes': top3_classes,
                'top3_probs': top3_probs,
                'high_confidence_classes': high_confidence_matches,
                'high_confidence_scores': high_confidence_scores,
                'class_names': [self.class_names[idx] for idx in high_confidence_matches],
                'top3_class_names': [[self.class_names[c] for c in row] for row in top3_classes],
            }
            
            predictions[folder_name] = folder_predictions
            
            self.logger.info(f"\n{folder_name}:")
            self.logger.info(f"  Total images: {len(images)}")
            self.logger.info(f"  High confidence matches (>={threshold:.0%}): {len(high_confidence_matches)}")
            self.logger.info(f"  Mean confidence: {np.mean(pred_confidences):.4f}")
            
            if len(high_confidence_matches) > 0:
                self.logger.info(f"  Matched classes: {folder_predictions['class_names']}")
            # Log individual top-3 for research analysis (always, so every
            # image gets a model verdict in the logs / dashboard)
            if folder_name in self.validation_files:
                for i, fname in enumerate(self.validation_files[folder_name]):
                    t3 = folder_predictions['top3_class_names'][i]
                    t3p = folder_predictions['top3_probs'][i]
                    self.logger.info(f"    [{fname}] Top-3: {list(zip(t3, [f'{p:.3f}' for p in t3p]))}")
        
        return predictions
    
    def analyze_civilization_link(self, predictions):
        """
        Analyze the strength of link between Indus and Keeladi civilizations
        
        Args:
            predictions: Dictionary of predictions from validation sets
            
        Returns:
            Analysis results dictionary
        """
        self.logger.info("Analyzing Indus-Keeladi civilization link...")
        analysis = {
            'total_keeladi_images': 0,
            'total_high_confidence_matches': 0,
            'match_rate': 0.0,
            'mean_confidence': 0.0,
            'direct_matches': {},
            'most_common_indus_signs': {},
            'known_pair': {},          # folder -> top-1 correctness vs expected sign
            'known_pair_correct': 0,
            'known_pair_total': 0,
            'unmapped_folders': [],    # match folders without an expected-class mapping
            'negative_rejection': {},  # control folders -> rejection/false-match stats
        }
        
        all_confidences = []
        
        # Count total images and matches
        for folder_name, pred_data in predictions.items():
            num_images = len(pred_data['all_predictions'])
            num_matches = len(pred_data['high_confidence_classes'])
            mean_conf = float(np.mean(pred_data['all_confidences']))
            all_confidences.extend(pred_data['all_confidences'])
            
            analysis['total_keeladi_images'] += num_images
            analysis['total_high_confidence_matches'] += num_matches
            
            # Track direct matches
            if folder_name.startswith('match_Indus_'):
                analysis['direct_matches'][folder_name] = {
                    'images': num_images,
                    'matches': num_matches,
                    'match_rate': num_matches / num_images if num_images > 0 else 0,
                    'mean_confidence': mean_conf
                }
                # Honest top-1 correctness against the expected correspondence
                expected = EXPECTED_MATCH_MAP.get(folder_name)
                if expected is None:
                    analysis['unmapped_folders'].append(folder_name)
                elif expected in self.class_names:
                    expected_idx = self.class_names.index(expected)
                    correct = int(np.sum(pred_data['all_predictions'] == expected_idx))
                    analysis['known_pair'][folder_name] = {
                        'expected': expected,
                        'images': num_images,
                        'correct': correct,
                    }
                    analysis['known_pair_correct'] += correct
                    analysis['known_pair_total'] += num_images
                else:
                    analysis['unmapped_folders'].append(folder_name)
            
            # Control folders: correct behaviour is REJECTION, not a match
            if self._is_negative_folder(folder_name):
                rejected = int(np.sum(pred_data['is_rejection']))
                false_matches = int(np.sum(
                    (pred_data['all_confidences'] >= 0.5)
                    & ~pred_data['is_rejection']))
                analysis['negative_rejection'][folder_name] = {
                    'images': num_images,
                    'rejected': rejected,
                    'rejection_rate': rejected / num_images if num_images else 0.0,
                    'false_matches': false_matches,
                    'false_match_rate': false_matches / num_images if num_images else 0.0,
                }
        
        # Calculate overall match rate
        if analysis['total_keeladi_images'] > 0:
            analysis['match_rate'] = analysis['total_high_confidence_matches'] / analysis['total_keeladi_images']
            analysis['mean_confidence'] = float(np.mean(all_confidences)) if all_confidences else 0.0
        
        # Find most commonly matched Indus signs (using all predictions, not just high confidence)
        all_predicted_classes = []
        for pred_data in predictions.values():
            all_predicted_classes.extend(pred_data['all_predictions'])
        
        if all_predicted_classes:
            unique_classes, counts = np.unique(all_predicted_classes, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            
            for idx in sorted_indices[:15]:  # Top 15
                class_idx = unique_classes[idx]
                class_name = self.class_names[class_idx]
                analysis['most_common_indus_signs'][class_name] = int(counts[idx])
        
        self.logger.info(f"  Total images: {analysis['total_keeladi_images']}")
        self.logger.info(f"  Match rate: {analysis['match_rate']:.2%}")
        self.logger.info(f"  Mean confidence: {analysis['mean_confidence']:.4f}")
        self.logger.info(f"  Unique Indus signs matched: {len(analysis['most_common_indus_signs'])}")
        
        # Honest metrics log
        if analysis['known_pair_total']:
            self.logger.info(
                f"  Known-pair top-1 accuracy: "
                f"{analysis['known_pair_correct']}/{analysis['known_pair_total']}")
        for folder, kp in analysis['known_pair'].items():
            self.logger.info(
                f"    {folder}: expected {kp['expected']} -> "
                f"{kp['correct']}/{kp['images']} correct")
        if analysis['unmapped_folders']:
            self.logger.info(
                f"  Unmapped match folders (expected sign not in training set): "
                f"{analysis['unmapped_folders']}")
        for folder, nr in analysis['negative_rejection'].items():
            self.logger.info(
                f"    control {folder}: rejected {nr['rejected']}/{nr['images']} "
                f"({nr['rejection_rate']:.0%}), false matches {nr['false_matches']}")
        
        return analysis
    
    def generate_report(self, predictions, analysis, output_dir):
        """
        Generate comprehensive evaluation report with text and visualizations
        
        Args:
            predictions: Dictionary of predictions
            analysis: Analysis results
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Text report
        report_path = output_path / "keeladi_evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("   INDUS-KEELADI CIVILIZATION LINK EVALUATION REPORT\n")
            f.write("   CNN-Based Pattern Matching Analysis\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Keeladi images analyzed: {analysis['total_keeladi_images']}\n")
            f.write(f"Total high-confidence Indus matches: {analysis['total_high_confidence_matches']}\n")
            f.write(f"Overall match rate: {analysis['match_rate']:.2%}\n")
            f.write(f"Mean prediction confidence: {analysis['mean_confidence']:.4f}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("DIRECT MATCH ANALYSIS (Expected Archaeological Correspondences)\n")
            f.write("-" * 70 + "\n\n")
            
            for match_name, stats in analysis['direct_matches'].items():
                f.write(f"\n{match_name.upper()}:\n")
                f.write(f"  Images tested:      {stats['images']}\n")
                f.write(f"  High-conf matches:  {stats['matches']}\n")
                f.write(f"  Match rate:         {stats['match_rate']:.2%}\n")
                f.write(f"  Mean confidence:    {stats['mean_confidence']:.4f}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("HONEST METRICS (what the numbers above actually mean)\n")
            f.write("-" * 70 + "\n\n")
            
            f.write("Known-pair top-1 accuracy (did the model predict the EXPECTED sign?):\n")
            if analysis['known_pair_total']:
                f.write(f"  Overall: {analysis['known_pair_correct']}/"
                        f"{analysis['known_pair_total']}\n")
            else:
                f.write("  Overall: no mapped known pairs\n")
            for folder, kp in analysis['known_pair'].items():
                f.write(f"  {folder}: expected {kp['expected']} -> "
                        f"{kp['correct']}/{kp['images']} correct\n")
            if analysis['unmapped_folders']:
                f.write("  Unmapped folders (expected sign has no training class;\n"
                        "  fill EXPECTED_MATCH_MAP in src/evaluate.py from your\n"
                        "  notebook sources):\n")
                for folder in analysis['unmapped_folders']:
                    f.write(f"    - {folder}\n")
            
            f.write("\nControl folders (correct answer = REJECT, not match):\n")
            if analysis['negative_rejection']:
                for folder, nr in analysis['negative_rejection'].items():
                    f.write(f"  {folder}: rejected {nr['rejected']}/{nr['images']} "
                            f"({nr['rejection_rate']:.0%}), "
                            f"false matches {nr['false_matches']} "
                            f"({nr['false_match_rate']:.0%})\n")
            else:
                f.write("  none\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("TOP 15 MOST FREQUENTLY MATCHED INDUS SIGNS\n")
            f.write("-" * 70 + "\n\n")
            
            for i, (sign_name, count) in enumerate(analysis['most_common_indus_signs'].items(), 1):
                f.write(f"  {i:2d}. {sign_name:<40s}: {count:3d} matches\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write("DETAILED PER-IMAGE PREDICTIONS (Top-3)\n")
            f.write("-" * 70 + "\n\n")
            
            for folder_name, pred_data in predictions.items():
                f.write(f"\n[{folder_name}]\n")
                if folder_name in getattr(self, 'validation_files', {}):
                    for i, fname in enumerate(self.validation_files[folder_name]):
                        t3_names = pred_data['top3_class_names'][i]
                        t3_probs = pred_data['top3_probs'][i]
                        f.write(f"  {fname}:\n")
                        for rank, (name, prob) in enumerate(zip(t3_names, t3_probs), 1):
                            bar = '#' * int(prob * 40)
                            f.write(f"    #{rank}: {name:<40s} {prob:.3f} {bar}\n")
        
        self.logger.info(f"Text report saved to {report_path}")
        
        # Visualizations
        self._plot_match_statistics(analysis, output_path)
        self._plot_confidence_distribution(predictions, output_path)

        # ── Graffiti gallery + side-by-side comparisons ──────────────────
        self.logger.info("Generating Keeladi graffiti gallery visualizations...")
        self._plot_graffiti_gallery(
            predictions,
            output_path,
            folder_filter=['general_keeladi_graffiti', 'match_Indus_*', 'tamil_brahmi_*']
        )
    
    def run_decoding(self, output_dir):
        """
        Parallel dual-script inscription decoding stage.

        Every Atan letter and every graffiti is read against BOTH scripts
        in parallel (Tamil-Brahmi reference alphabet + Indus CNN), each
        identification shown together with its lexicon text annotation.
        Pot outlines / sherd ends are accounted for (reported) but never
        classified, so they cannot cause false comparisons.
        """
        from src.decoding import InscriptionDecoder, KEELADI_NAMES

        output_dir = Path(output_dir)
        decode_dir = output_dir / "decoded"
        decode_dir.mkdir(parents=True, exist_ok=True)

        decoder = InscriptionDecoder(self.data_dir, self.normalizer,
                                     classifier=self.classifier,
                                     class_names=self.class_names)
        lines = ["PARALLEL DECODING - Tamil-Brahmi vs Indus",
                 "=" * 70,
                 "B: = Tamil-Brahmi read (deciphered, real readings)",
                 "I: = Indus read (meanings from the project lexicon data/lexicon.json)",
                 "",
                 "--- Atan potsherds (multi-letter inscriptions) ---"]

        def ann(text, meaning):
            return f"{text}" + (f" '{meaning}'" if meaning else "")

        # 1. Multi-character Tamil-Brahmi inscriptions
        tb_dir = self.data_dir / "processed" / "val" / "tamil_brahmi"
        insc_folders = sorted(tb_dir.glob("inscriptions_*")) if tb_dir.exists() else []
        for folder in insc_folders:
            for img in sorted(self._list_image_files(folder)):
                result = decoder.decode_inscription(img)
                outline_note = ("outline accounted for, excluded from "
                                "classification" if result["outline_present"]
                                else "no outline")
                lines.append(f"{img.name} [{outline_note}]")
                self.logger.info(f"DECODED {img.name}: "
                                 f"{len(result['segments'])} letters, "
                                 f"outline_present={result['outline_present']}")
                if not result["segments"]:
                    lines.append("  no letter segments (inscription may be "
                                 "fused to the outline - manual ROI needed)")
                    continue
                decoder.visualize_decoding(
                    img, result, decode_dir / f"{img.stem}_annotated.png")
                for i, seg in enumerate(result["segments"], 1):
                    b_txt = ann(f"{seg['transliteration'] or seg['letter_id']}"
                                f" ({seg['score']:.2f})", seg['meaning'])
                    i_txt = "no Indus read"
                    if seg.get("indus_top3"):
                        t = seg["indus_top3"][0]
                        i_txt = ann(f"{t['class']} (p={t['prob']:.2f})",
                                    t['meaning'])
                    lines.append(f"  L{i}: B: {b_txt} | I: {i_txt}")
                lines.append(f"  Brahmi reading: {result['brahmi_reading']}")
                lines.append(f"  Indus reading:  {result['indus_reading']}")
                nlp = decoder.nlp_decode(result, corpus_key=folder.name)
                lines.append("  NLP DECODE (CNN letters + NLP composition):")
                if nlp["corpus"]:
                    reading, meaning = nlp["corpus"]
                    lines.append(f"    decoded name: '{reading}' - {meaning}")
                if nlp["name_match"]:
                    nm, nr = nlp["name_match"]
                    lines.append(f"    name match: {nm} - "
                                 f"{KEELADI_NAMES[nm]} (similarity {nr:.0%})")
                lines.append(f"    letter reading: {result['brahmi_reading']} "
                             f"(composed: {nlp['word'] or '-'})")
                lines.append(f"    Indus gloss: {nlp['indus_gloss'] or '-'}")

        # 2. Single-symbol Keeladi graffiti, read against both scripts
        lines += ["", "--- Keeladi graffiti (single symbols, both reads) ---"]
        val_dir = self.data_dir / "processed" / "val" / "keeladi"
        graffiti_folders = sorted(p for p in val_dir.iterdir() if p.is_dir()) \
            if val_dir.exists() else []
        for folder in graffiti_folders:
            for img in sorted(self._list_image_files(folder)):
                processed = self.normalizer.process_image(img)
                r = decoder.decode_graffiti(processed)
                letter_id, score = r["brahmi"]
                b_txt = ann(f"{r['brahmi_translit'] or letter_id} "
                            f"({score:.2f})", r["brahmi_meaning"]) \
                    if letter_id else "no Brahmi read"
                i_parts = []
                for t in r["indus_top3"]:
                    i_parts.append(ann(f"{t['class']} (p={t['prob']:.2f})",
                                       t['meaning']))
                lines.append(f"{folder.name}/{img.name}:")
                lines.append(f"  B: {b_txt}")
                lines.append(f"  I: {' | '.join(i_parts)}")

        report_path = decode_dir / "decoded_readings.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self.logger.info(f"Decoded readings saved to {report_path}")
        return report_path

    def generate_known_pair_comparison(self, output_dir):
        """
        Notebook-style side-by-side figure: Keeladi graffiti (left) vs
        INDUS sign (right) with a shape MATCH PERCENTAGE per row - the
        same layout as the researcher's notebook comparison chart.

        For every curated Indus reference image the best-matching Keeladi
        potsherd is found automatically (dilation-tolerant IoU between the
        reference glyph and the potsherd's inner-letter glyph, outline
        excluded), then the pairs + percentages are rendered and saved as
        known_pair_comparison.png / known_pair_scores.json.
        """
        import cv2
        import json as _json
        import re
        from src.decoding import InscriptionDecoder

        output_dir = Path(output_dir)
        matched_dir = self.data_dir / "processed" / "train" / "indus_matched"
        val_keeladi = self.data_dir / "processed" / "val" / "keeladi"
        if not matched_dir.exists() or not val_keeladi.exists():
            return None

        decoder = InscriptionDecoder(self.data_dir, self.normalizer)

        # candidate pool: every Keeladi val image (match folders + graffiti)
        candidates = []
        for folder in sorted(val_keeladi.iterdir()):
            if folder.is_dir():
                candidates.extend(sorted(self._list_image_files(folder)))

        def inner_glyphs(path):
            """Per-letter glyphs of a potsherd (outline excluded), 64x64.

            The Indus reference is compared against each letter segment
            separately and the best local match is kept, so extra strokes
            elsewhere on the sherd cannot dilute the score."""
            segments, _outline = decoder.segment_inscription(path)
            return [s["glyph"] for s in segments]

        def ref_glyph(path):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return None
            binary = decoder._binarize(gray)
            n, _l, cstats, _c = cv2.connectedComponentsWithStats(binary, 8)
            if n < 2:
                return None
            i_max = 1 + int(np.argmax(cstats[1:, 4]))
            x, y, bw, bh, _a = cstats[i_max]
            return decoder._crop_to_bbox(binary, x, y, bw, bh)

        def cov(a, b):
            """Bidirectional stroke coverage: how much of each drawing's
            strokes are present in the (dilated) other one."""
            k = np.ones((3, 3), np.uint8)
            A = (a > 0).astype(np.uint8)
            B = (b > 0).astype(np.uint8)
            na, nb = np.count_nonzero(A), np.count_nonzero(B)
            if not na or not nb:
                return 0.0
            Ad = cv2.dilate(A, k, iterations=2)
            Bd = cv2.dilate(B, k, iterations=2)
            ca = np.count_nonzero(A & Bd) / na
            cb = np.count_nonzero(B & Ad) / nb
            return 100.0 * (ca + cb) / 2.0

        def shape_pct(rg, glyph_list):
            return max([cov(rg, g) for g in glyph_list], default=0.0)

        cand_glyphs = [(c, inner_glyphs(c)) for c in candidates]
        cand_glyphs = [(c, g) for c, g in cand_glyphs if g]

        used = set()
        rows = []
        for sub in sorted(matched_dir.iterdir()):
            if not sub.is_dir():
                continue
            # pair each reference with its expected Keeladi match folder
            # (the notebook pairing); variants / missing folders fall back
            # to the best not-yet-used candidate so no sherd repeats
            m = re.search(r"P(\d+)", sub.name)
            val_dir = val_keeladi / f"match_Indus_{m.group(1)}" if m else None
            expected = sorted(self._list_image_files(val_dir)) \
                if val_dir and val_dir.exists() else []
            for ref in sorted(self._list_image_files(sub)):
                rg = ref_glyph(ref)
                if rg is None:
                    continue
                pool = [cg for cg in cand_glyphs
                        if cg[0] in expected and cg[0].name not in used]
                if not pool:
                    pool = [cg for cg in cand_glyphs
                            if cg[0].name not in used] or cand_glyphs
                best_cand, best_pct = max(
                    ((cg, shape_pct(rg, gl)) for cg, gl in pool),
                    key=lambda t: t[1])
                used.add(best_cand.name)
                rows.append({"class": sub.name, "indus": ref.name,
                             "keeladi": best_cand.name,
                             "match_pct": round(best_pct, 1)})
        order = {"225": 0, "307": 1, "365": 2, "318": 3, "318b": 4}
        rows.sort(key=lambda r: order.get(Path(r["indus"]).stem, 99))
        if not rows:
            return None

        # notebook-style two-column figure with a percentage column
        fig, axes = plt.subplots(len(rows), 3, figsize=(7, 3 * len(rows)),
                                 gridspec_kw={"width_ratios": [1, 1, 0.45]})
        if len(rows) == 1:
            axes = np.array([axes])
        axes[0, 0].set_title("Keeladi graffiti", fontweight="bold")
        axes[0, 1].set_title("INDUS sign", fontweight="bold")
        axes[0, 2].set_title("Match", fontweight="bold")
        for r, row in enumerate(rows):
            cand_path = next(c for c in candidates if c.name == row["keeladi"])
            ref_path = matched_dir / row["class"] / row["indus"]
            for ax, path in ((axes[r, 0], cand_path), (axes[r, 1], ref_path)):
                img = cv2.imread(str(path))
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.axis("off")
            axes[r, 2].axis("off")
            axes[r, 2].text(0.5, 0.5, f'{row["match_pct"]:.0f}%',
                            ha="center", va="center",
                            fontsize=20, fontweight="bold")
        fig.tight_layout()
        png = output_dir / "known_pair_comparison.png"
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        with open(output_dir / "known_pair_scores.json", "w",
                  encoding="utf-8") as f:
            _json.dump(rows, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Known-pair comparison figure saved to {png} "
                         f"({len(rows)} rows)")
        return png

    def _plot_match_statistics(self, analysis, output_dir):
        """Create visualization of match statistics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            fig.suptitle('Indus-Keeladi Civilization Link Analysis', fontsize=14, fontweight='bold')
            
            # Direct match rates
            if analysis['direct_matches']:
                match_names = list(analysis['direct_matches'].keys())
                match_rates = [stats['match_rate'] for stats in analysis['direct_matches'].values()]
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(match_names)))
                
                bars = axes[0, 0].bar(match_names, match_rates, color=colors)
                axes[0, 0].set_title('Direct Match Rates (Expected Correspondences)')
                axes[0, 0].set_ylabel('Match Rate')
                axes[0, 0].set_ylim(0, 1.1)
                axes[0, 0].tick_params(axis='x', rotation=30, labelsize=8)
                for bar, rate in zip(bars, match_rates):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                   f'{rate:.0%}', ha='center', va='bottom', fontsize=9)
            
            # Most common signs
            if analysis['most_common_indus_signs']:
                sign_names = list(analysis['most_common_indus_signs'].keys())[:10]
                sign_counts = list(analysis['most_common_indus_signs'].values())[:10]
                
                axes[0, 1].barh(sign_names[::-1], sign_counts[::-1], 
                               color=plt.cm.plasma(np.linspace(0.4, 0.9, len(sign_names))))
                axes[0, 1].set_title('Top 10 Most Frequently Matched Indus Signs')
                axes[0, 1].set_xlabel('Number of Predictions')
                axes[0, 1].tick_params(axis='y', labelsize=7)
            
            # Overall statistics pie
            if analysis['total_keeladi_images'] > 0:
                labels = ['High-Conf Matches', 'Lower-Conf Predictions']
                sizes = [analysis['total_high_confidence_matches'],
                        analysis['total_keeladi_images'] - analysis['total_high_confidence_matches']]
                colors_pie = ['#2ecc71', '#e67e22']
                if sum(sizes) > 0:
                    wedges, texts, autotexts = axes[1, 0].pie(sizes, labels=labels, colors=colors_pie,
                                                               autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Prediction Confidence Distribution')
            
            # Summary text
            axes[1, 1].axis('off')
            summary_text = (
                "=======================================\n"
                "       EVALUATION SUMMARY\n"
                "=======================================\n\n"
                f"  Total Keeladi Images:  {analysis['total_keeladi_images']:>5}\n"
                f"  High-Conf Matches:     {analysis['total_high_confidence_matches']:>5}\n"
                f"  Match Rate:            {analysis['match_rate']:>10.1%}\n"
                f"  Mean Confidence:       {analysis['mean_confidence']:>10.3f}\n\n"
                f"  Direct Match Folders:  {len(analysis['direct_matches']):>5}\n"
                f"  Unique Indus Signs:    {len(analysis['most_common_indus_signs']):>5}\n\n"
                "=======================================\n"
                "  Research Gap Addressed:\n"
                "  - Scaled comparison from 4 -> 100%\n"
                "    of Keeladi graffiti dataset\n"
                "  - Objective mathematical proof\n"
                "    of visual resemblance\n"
                "  - Evolutionary feature mapping\n"
                "======================================="
            )
            axes[1, 1].text(0.05, 0.95, summary_text, fontsize=9,
                           verticalalignment='top', family='monospace',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.7))
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plot_path = output_dir / "match_statistics.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Statistics plot saved to {plot_path}")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting statistics: {e}")
    
    def _plot_confidence_distribution(self, predictions, output_dir):
        """Plot distribution of prediction confidences across all datasets"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            all_confs = []
            folder_labels = []
            for folder_name, pred_data in predictions.items():
                confs = pred_data['all_confidences']
                all_confs.extend(confs)
                folder_labels.extend([folder_name] * len(confs))
            
            if all_confs:
                bins = np.linspace(0, 1, 21)
                ax.hist(all_confs, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
                ax.axvline(x=0.5, color='red', linestyle='--', label='50% Threshold')
                ax.axvline(x=0.7, color='orange', linestyle='--', label='70% Threshold')
                ax.axvline(x=np.mean(all_confs), color='green', linestyle='-', label=f'Mean ({np.mean(all_confs):.3f})')
                
                ax.set_title('Distribution of Prediction Confidences')
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Number of Predictions')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = output_dir / "confidence_distribution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Confidence distribution plot saved to {plot_path}")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting confidence distribution: {e}")

    def _plot_graffiti_gallery(self, predictions, output_dir, folder_filter=None, max_per_page=12):
        """
        Generate per-image gallery PNGs for Keeladi graffiti.
        Each subplot shows the graffiti image + Top-3 Indus sign predictions with probabilities.

        Args:
            predictions: Dictionary from predict_keeladi_matches()
            output_dir: Directory for gallery PNGs
            folder_filter: Optional list of folder keys to include (e.g. ['general_keeladi_graffiti', 'tamil_brahmi_*'])
            max_per_page: Max sherds per PNG page
        """
        try:
            import cv2
            import math

            validation_files = getattr(self, 'validation_files', {})
            all_graffiti_folders = []
            for folder_name in predictions.keys():
                if folder_filter is None:
                    all_graffiti_folders.append(folder_name)
                else:
                    matched = False
                    for pat in folder_filter:
                        if (pat.endswith('*') and folder_name.startswith(pat[:-1])) or folder_name == pat:
                            matched = True
                            break
                    if matched:
                        all_graffiti_folders.append(folder_name)

            train_sign_dir = self.data_dir / "processed" / "train" / "primary_core_signs"

            for folder_name in all_graffiti_folders:
                pred_data = predictions[folder_name]
                files_in_folder = validation_files.get(folder_name, [])
                n = len(pred_data['all_predictions'])
                if n == 0:
                    continue

                files_folder = None
                if folder_name.startswith('tamil_brahmi_'):
                    sub_key = folder_name[len('tamil_brahmi_'):]
                    files_folder = self.data_dir / "processed" / "val" / "tamil_brahmi" / sub_key
                elif folder_name.startswith('match_Indus_') or folder_name == 'general_keeladi_graffiti':
                    files_folder = self.data_dir / "processed" / "val" / "keeladi" / folder_name

                num_pages = math.ceil(n / max_per_page)

                for page in range(num_pages):
                    start = page * max_per_page
                    end = min(start + max_per_page, n)
                    count = end - start
                    cols = 3
                    rows = math.ceil(count / cols)
                    fig_w = 16
                    fig_h = rows * 5.2
                    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
                    fig.suptitle(f'Keeladi Graffiti Gallery — {folder_name} (Page {page+1}/{num_pages})',
                                 fontsize=15, fontweight='bold', color='#2c3e50')

                    flat_axes = axes.flatten()
                    for idx_in_page, i in enumerate(range(start, end)):
                        ax = flat_axes[idx_in_page]
                        fname = files_in_folder[i] if i < len(files_in_folder) else f'img_{i}.png'
                        img_path = files_folder / fname if files_folder else None

                        graffiti_img = None
                        if img_path and img_path.exists():
                            try:
                                img_bgr = cv2.imread(str(img_path))
                                if img_bgr is not None:
                                    graffiti_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            except Exception:
                                graffiti_img = None

                        if graffiti_img is not None:
                            ax.imshow(graffiti_img)
                        else:
                            ax.text(0.5, 0.5, f'(image not found)\n{fname}', ha='center', va='center', fontsize=8)
                            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

                        ax.set_xticks([]); ax.set_yticks([])

                        top3_names = pred_data['top3_class_names'][i]
                        top3_probs = pred_data['top3_probs'][i]
                        lines = []
                        for rank, (name, prob) in enumerate(zip(top3_names, top3_probs), 1):
                            pct = f'{prob:.1%}'
                            bar = '\u2588' * int(prob * 25)
                            lines.append(f'#{rank} {name[:30]}  {pct}  {bar}')
                        ax.set_title('\n'.join(lines), fontsize=7.5, loc='center',
                                     backgroundcolor='#f8f9fa', pad=6, family='monospace',
                                     color='#212529')

                    for j in range(idx_in_page + 1, len(flat_axes)):
                        flat_axes[j].axis('off')

                    fig.text(0.5, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}  |  '
                                        f'CNN trained on {len(self.class_names)} Indus Core Signs  |  '
                                        f'Page {page+1}/{num_pages}',
                             ha='center', fontsize=8, color='#6c757d', style='italic')
                    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

                    safe_folder = folder_name.replace('/', '_').replace('\\', '_')
                    gallery_path = output_dir / f"graffiti_gallery_{safe_folder}_page{page+1:02d}.png"
                    plt.savefig(gallery_path, dpi=140, bbox_inches='tight', facecolor='white')
                    self.logger.info(f"Gallery page saved to {gallery_path}")
                    plt.close()

                self._save_single_sherd_comparisons(folder_name, predictions, output_dir, train_sign_dir)

        except Exception as e:
            self.logger.error(f"Error generating graffiti gallery: {e}", exc_info=True)

    def _save_single_sherd_comparisons(self, folder_name, predictions, output_dir, train_sign_dir):
        """
        For each graffiti sherd, save a side-by-side comparison PNG:
        Left column = graffiti image, Right column = Top-3 predicted Indus signs images.
        """
        try:
            import cv2
            import math

            validation_files = getattr(self, 'validation_files', {})
            pred_data = predictions[folder_name]
            files_in_folder = validation_files.get(folder_name, [])
            n = len(pred_data['all_predictions'])
            if n == 0:
                return

            files_folder = None
            if folder_name.startswith('tamil_brahmi_'):
                sub_key = folder_name[len('tamil_brahmi_'):]
                files_folder = self.data_dir / "processed" / "val" / "tamil_brahmi" / sub_key
            elif folder_name.startswith('match_Indus_') or folder_name == 'general_keeladi_graffiti':
                files_folder = self.data_dir / "processed" / "val" / "keeladi" / folder_name

            sign_folder_lookup = {}
            if train_sign_dir.exists():
                for d in train_sign_dir.iterdir():
                    if d.is_dir():
                        sign_folder_lookup[d.name] = d

            max_rows = min(n, 10)
            total_pages = math.ceil(n / max_rows)
            for page in range(total_pages):
                start = page * max_rows
                end = min(start + max_rows, n)
                page_count = end - start
                fig, axes = plt.subplots(page_count, 2, figsize=(10, page_count * 2.7), squeeze=False)
                fig.suptitle(f'Graffiti ↔ Indus Sign Comparison — {folder_name} (Page {page+1}/{total_pages})',
                             fontsize=14, fontweight='bold', color='#1a237e')

                for local_i, i in enumerate(range(start, end)):
                    fname = files_in_folder[i] if i < len(files_in_folder) else f'img_{i}.png'
                    img_path = files_folder / fname if files_folder else None

                    # Left: Graffiti
                    ax_left = axes[local_i, 0]
                    graffiti_img = None
                    if img_path and img_path.exists():
                        try:
                            img_bgr = cv2.imread(str(img_path))
                            if img_bgr is not None:
                                graffiti_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        except Exception:
                            graffiti_img = None
                    if graffiti_img is not None:
                        ax_left.imshow(graffiti_img)
                        ax_left.set_title(f'Keeladi Graffiti\n{fname[:45]}', fontsize=9, pad=4)
                    else:
                        ax_left.text(0.5, 0.5, f'{fname}', ha='center', va='center', fontsize=8)
                        ax_left.set_xlim(0, 1); ax_left.set_ylim(0, 1)
                    ax_left.set_xticks([]); ax_left.set_yticks([])

                    # Right: Indus Top-3 montage (horizontal strip)
                    ax_right = axes[local_i, 1]
                    ax_right.set_xticks([]); ax_right.set_yticks([])
                    top3_names = pred_data['top3_class_names'][i]
                    top3_probs = pred_data['top3_probs'][i]

                    strip_axes = []
                    for rank in range(3):
                        sub = ax_right.inset_axes([rank * 0.33 + 0.01, 0.20, 0.31, 0.70])
                        strip_axes.append(sub)

                    for rank, (sub, name, prob) in enumerate(zip(strip_axes, top3_names, top3_probs)):
                        sign_path = None
                        if name in sign_folder_lookup:
                            imgs = sorted(sign_folder_lookup[name].glob("*.png")) + \
                                   sorted(sign_folder_lookup[name].glob("*.jpg"))
                            if imgs:
                                sign_path = imgs[0]
                        sign_img = None
                        if sign_path and sign_path.exists():
                            try:
                                sb = cv2.imread(str(sign_path))
                                if sb is not None:
                                    sign_img = cv2.cvtColor(sb, cv2.COLOR_BGR2RGB)
                            except Exception:
                                sign_img = None
                        if sign_img is not None:
                            sub.imshow(sign_img)
                        else:
                            sub.text(0.5, 0.5, name[:18], ha='center', va='center', fontsize=7, wrap=True)
                            sub.set_xlim(0, 1); sub.set_ylim(0, 1)
                        sub.set_xticks([]); sub.set_yticks([])
                        sub.set_title(f'#{rank+1}  {prob:.0%}\n{name[:22]}',
                                      fontsize=7.5, pad=2,
                                      backgroundcolor=['#d4efdf', '#fff3cd', '#f8d7da'][rank], wrap=True)

                fig.text(0.5, 0.01, f'Keeladi Graffiti compared against {len(self.class_names)} Indus signs | '
                                   f'Page {page+1}/{total_pages}',
                         ha='center', fontsize=8, color='#6c757d')
                plt.tight_layout(rect=[0, 0.035, 1, 0.95])

                safe_folder = folder_name.replace('/', '_').replace('\\', '_')
                comp_path = output_dir / f"graffiti_vs_indus_{safe_folder}_page{page+1:02d}.png"
                plt.savefig(comp_path, dpi=140, bbox_inches='tight', facecolor='white')
                self.logger.info(f"Comparison page saved to {comp_path}")
                plt.close()

        except Exception as e:
            self.logger.error(f"Error saving single sherd comparisons: {e}", exc_info=True)


def main():
    """Main evaluation pipeline"""
    
    # Set paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "indus_classifier.keras"
    data_dir = project_root / "data"
    output_dir = project_root / "models" / "evaluation_results"
    log_dir = data_dir / "results" / "logs"
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("=" * 60)
    logger.info("INDUS-KEELADI CIVILIZATION LINK EVALUATION STARTED")
    logger.info("=" * 60)
    
    try:
        # Check model exists
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}. Run training first.")
            sys.exit(1)
        
        # Initialize evaluator
        logger.info("Initializing Keeladi evaluator...")
        evaluator = KeeladiEvaluator(model_path, data_dir)
        
        # Load validation data
        logger.info("Loading Keeladi validation sets...")
        validation_data = evaluator.load_keeladi_validation_set()
        
        if not validation_data:
            logger.error("No validation images found. Check val/keeladi and val/tamil_brahmi directories.")
            sys.exit(1)
        
        # Predict matches - use 0.5 threshold for research discovery
        logger.info("Running predictions with 50% confidence threshold...")
        predictions = evaluator.predict_keeladi_matches(validation_data, threshold=0.5)
        
        # Analyze civilization link
        logger.info("Analyzing civilization link metrics...")
        analysis = evaluator.analyze_civilization_link(predictions)
        
        # Generate report
        logger.info("Generating evaluation reports and visualizations...")
        evaluator.generate_report(predictions, analysis, output_dir)
        
        # Decode inscriptions (segmentation + lexicon readings)
        logger.info("Running inscription decoding...")
        evaluator.run_decoding(output_dir)

        # Notebook-style Keeladi-vs-Indus comparison with match %
        logger.info("Generating known-pair comparison figure...")
        evaluator.generate_known_pair_comparison(output_dir)
        
        logger.info("=" * 60)
        logger.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Reports saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
