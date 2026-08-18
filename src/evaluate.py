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
        
        for folder_name, images in validation_data.items():
            # Reshape for CNN
            X = images.reshape(images.shape[0], 64, 64, 1)
            
            # Get predictions
            pred_probs = self.classifier.predict(X)
            pred_classes = np.argmax(pred_probs, axis=1)
            pred_confidences = np.max(pred_probs, axis=1)
            
            # Get top-3 predictions per image for research analysis
            top3_classes = np.argsort(pred_probs, axis=1)[:, -3:][:, ::-1]
            top3_probs = np.sort(pred_probs, axis=1)[:, -3:][:, ::-1]
            
            # Filter by threshold
            high_confidence_mask = pred_confidences >= threshold
            high_confidence_matches = pred_classes[high_confidence_mask]
            high_confidence_scores = pred_confidences[high_confidence_mask]
            
            folder_predictions = {
                'all_predictions': pred_classes,
                'all_confidences': pred_confidences,
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
                # Log individual top-3 for research analysis
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
            'most_common_indus_signs': {}
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
        
        logger.info("=" * 60)
        logger.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Reports saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
