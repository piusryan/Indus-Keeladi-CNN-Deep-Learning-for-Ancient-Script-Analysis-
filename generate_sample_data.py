"""
Generate sample dataset images for demonstration
Creates synthetic Indus script and Keeladi graffiti images
"""

import numpy as np
import cv2
from pathlib import Path
import random


def generate_synthetic_sign(output_path, sign_type="basic"):
    """Generate a synthetic sign image"""
    img = np.zeros((64, 64), dtype=np.uint8)
    
    if sign_type == "cross":
        # Cross shape
        cv2.line(img, (20, 20), (44, 44), 255, 3)
        cv2.line(img, (44, 20), (20, 44), 255, 3)
    elif sign_type == "circle":
        # Circle shape
        cv2.circle(img, (32, 32), 15, 255, 2)
    elif sign_type == "lines":
        # Parallel lines
        cv2.line(img, (15, 20), (15, 44), 255, 2)
        cv2.line(img, (25, 20), (25, 44), 255, 2)
        cv2.line(img, (35, 20), (35, 44), 255, 2)
    elif sign_type == "triangle":
        # Triangle
        pts = np.array([[32, 15], [15, 45], [49, 45]], np.int32)
        cv2.polylines(img, [pts], True, 255, 2)
    elif sign_type == "square":
        # Square
        cv2.rectangle(img, (18, 18), (46, 46), 255, 2)
    else:
        # Random basic shape
        shapes = ["cross", "circle", "lines", "triangle", "square"]
        return generate_synthetic_sign(output_path, random.choice(shapes))
    
    # Add some noise for realism
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    cv2.imwrite(str(output_path), img)


def create_sample_dataset():
    """Create sample dataset for demonstration"""
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "processed" / "train" / "primary_core_signs"
    val_dir = project_root / "data" / "processed" / "val_keeladi"
    
    sign_types = ["cross", "circle", "lines", "triangle", "square"]
    
    # Create training data for first 10 classes (for demo speed)
    for i in range(1, 11):
        class_name = f"sign_{i:02d}_P13_Man" if i == 1 else f"sign_{i:02d}_P{60+i*5}"
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            # Use existing folder names
            existing_classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            if i <= len(existing_classes):
                class_dir = data_dir / existing_classes[i-1]
            else:
                continue
        
        print(f"Creating samples for {class_dir.name}")
        
        # Create 5 sample images per class
        for j in range(5):
            sign_type = sign_types[i % len(sign_types)]
            output_path = class_dir / f"sample_{j}.png"
            generate_synthetic_sign(output_path, sign_type)
    
    # Create validation data
    print("Creating validation data")
    for match_folder in ["match_Indus_225", "match_Indus_307", "match_Indus_365", "match_Indus_318"]:
        match_dir = val_dir / match_folder
        if match_dir.exists():
            for j in range(3):
                sign_type = sign_types[random.randint(0, len(sign_types)-1)]
                output_path = match_dir / f"keeladi_{j}.png"
                generate_synthetic_sign(output_path, sign_type)
    
    # Create general Keeladi graffiti
    general_dir = val_dir / "general_keeladi_graffiti"
    if general_dir.exists():
        for j in range(5):
            sign_type = sign_types[random.randint(0, len(sign_types)-1)]
            output_path = general_dir / f"graffiti_{j}.png"
            generate_synthetic_sign(output_path, sign_type)
    
    print("Sample dataset creation complete!")


if __name__ == "__main__":
    create_sample_dataset()
