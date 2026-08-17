"""
Run Dataset Analysis
Executes the dataset analysis notebook content as a Python script
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Set paths
project_root = Path(__file__).parent
data_dir = project_root / "data" / "processed" / "train" / "primary_core_signs"

print(f"Project root: {project_root}")
print(f"Data directory: {data_dir}")

# Get all sign classes
sign_classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
print(f"Total sign classes: {len(sign_classes)}")
print(f"\nSign classes:")
for i, sign_class in enumerate(sign_classes, 1):
    print(f"  {i}. {sign_class}")

# Count images per class
class_counts = {}
for sign_class in sign_classes:
    class_dir = data_dir / sign_class
    image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
    class_counts[sign_class] = len(image_files)

# Create DataFrame for analysis
df_counts = pd.DataFrame(list(class_counts.items()), columns=['Sign_Class', 'Image_Count'])
df_counts = df_counts.sort_values('Image_Count', ascending=False)

print("\nImage counts per class:")
print(df_counts.to_string(index=False))

# Visualize Class Distribution
plt.figure(figsize=(15, 8))
plt.bar(range(len(df_counts)), df_counts['Image_Count'])
plt.xlabel('Sign Class')
plt.ylabel('Number of Images')
plt.title('Distribution of Images Across Sign Classes')
plt.xticks(range(len(df_counts)), df_counts['Sign_Class'], rotation=90, fontsize=8)
plt.tight_layout()

# Save the plot
output_dir = project_root / "notebooks" / "analysis_results"
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / "class_distribution.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")
plt.show()

print(f"\nTotal images in dataset: {df_counts['Image_Count'].sum()}")
print(f"Average images per class: {df_counts['Image_Count'].mean():.2f}")
print(f"Min images per class: {df_counts['Image_Count'].min()}")
print(f"Max images per class: {df_counts['Image_Count'].max()}")

# Save analysis results
df_counts.to_csv(output_dir / "class_counts.csv", index=False)
print(f"Analysis results saved to: {output_dir}")

print("\n" + "="*60)
print("DATASET ANALYSIS SUMMARY")
print("="*60)
print("1. Dataset Overview: Distribution of images across 40 primary core signs")
print("2. Allographic Analysis: Understanding of natural variations in sign appearance")
print("3. Variation Metrics: Quantitative measures of sign complexity and variability")
print("4. Augmentation Strategy: Data augmentation recommendations to improve model robustness")
print("5. Quality Assessment: Identification of classes needing more samples or attention")
print("\nThe analysis ensures the CNN will be trained to recognize the underlying")
print("alphabetic sign despite variations in engraving style, material, and")
print("artistic interpretation - crucial for matching Keeladi graffiti to the Indus alphabet.")
