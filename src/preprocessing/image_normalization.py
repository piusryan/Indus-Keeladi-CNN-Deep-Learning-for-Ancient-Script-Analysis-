"""
Image Normalization Module for Indus-Keeladi CNN Project
Handles grayscale conversion, resizing, and standardization of sign images
"""

import cv2
import numpy as np
from pathlib import Path


class ImageNormalizer:
    """Normalizes Indus script and Keeladi graffiti images for CNN training"""
    
    def __init__(self, target_size=(64, 64)):
        """
        Initialize normalizer with target image size
        
        Args:
            target_size: Tuple of (height, width) for output images
        """
        self.target_size = target_size
    
    def load_image(self, image_path):
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array of image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def convert_to_grayscale(self, image):
        """
        Convert RGB/BGR image to grayscale
        
        Args:
            image: Input image array
            
        Returns:
            Grayscale image array
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def resize_image(self, image):
        """
        Resize image to target dimensions
        
        Args:
            image: Input image array
            
        Returns:
            Resized image array
        """
        return cv2.resize(image, (self.target_size[1], self.target_size[0]))
    
    def normalize_pixel_values(self, image):
        """
        Normalize pixel values to [0, 1] range
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image array
        """
        return image.astype(np.float32) / 255.0
    
    def apply_threshold(self, image, threshold=127):
        """
        Apply binary thresholding for better contrast
        
        Args:
            image: Grayscale input image
            threshold: Threshold value for binarization
            
        Returns:
            Thresholded binary image
        """
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    def process_image(self, image_path, apply_threshold=False):
        """
        Complete processing pipeline for a single image
        
        Args:
            image_path: Path to input image
            apply_threshold: Whether to apply binary thresholding
            
        Returns:
            Processed and normalized image array
        """
        image = self.load_image(image_path)
        image = self.convert_to_grayscale(image)
        
        if apply_threshold:
            image = self.apply_threshold(image)
        
        image = self.resize_image(image)
        image = self.normalize_pixel_values(image)
        
        return image
    
    def process_directory(self, input_dir, output_dir, apply_threshold=False):
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            apply_threshold: Whether to apply binary thresholding
            
        Returns:
            Number of images processed
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in image_extensions:
                try:
                    processed = self.process_image(image_file, apply_threshold)
                    output_file = output_path / image_file.name
                    
                    # Save as normalized float array or convert back to uint8
                    cv2.imwrite(str(output_file), (processed * 255).astype(np.uint8))
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        
        return processed_count


if __name__ == "__main__":
    # Example usage
    normalizer = ImageNormalizer(target_size=(64, 64))
    print("Image Normalizer initialized")
