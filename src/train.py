"""
Main Training Script for Indus-Keeladi CNN Project
Trains the classifier on Indus script signs and prepares for Keeladi validation
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.image_normalization import ImageNormalizer
from src.models.indus_classifier_cnn import IndusClassifierCNN


def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class IndusKeeladiTrainer:
    """
    Main training pipeline for Indus script classification
    """
    
    def __init__(self, data_dir, model_dir, augment_factor=20):
        """
        Initialize trainer
        
        Args:
            data_dir: Root directory for data
            model_dir: Directory to save trained models
            augment_factor: How many augmented copies to generate per original image
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.augment_factor = augment_factor
        
        self.normalizer = ImageNormalizer(target_size=(64, 64))
        self.classifier = None
        self.class_names = []
        self.data_augmentation = self._build_augmentation_pipeline()
    
    def _build_augmentation_pipeline(self):
        """Build Keras data augmentation pipeline for small datasets"""
        return keras.Sequential([
            keras.layers.RandomRotation(0.15),
            keras.layers.RandomZoom(0.1, 0.1),
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomContrast(0.2),
            keras.layers.GaussianNoise(0.05),
        ])
    
    def _augment_dataset(self, X, y):
        """
        Augment the dataset by generating augmented copies of each image
        
        Args:
            X: Input images (N, H, W, C)
            y: Labels (N,)
            
        Returns:
            Augmented X and y arrays
        """
        logger = logging.getLogger(__name__)
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            # Keep original image
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            
            # Generate augmented copies
            img = np.expand_dims(X[i], axis=0)
            for j in range(self.augment_factor):
                aug_img = self.data_augmentation(img, training=True)[0].numpy()
                augmented_X.append(aug_img)
                augmented_y.append(y[i])
        
        X_aug = np.array(augmented_X)
        y_aug = np.array(augmented_y)
        
        logger.info(f"Dataset augmented: {len(X)} -> {len(X_aug)} samples ({self.augment_factor}x factor)")
        return X_aug, y_aug
    
    def _safe_train_val_split(self, X, y, test_size=0.2):
        """
        Safely split data avoiding stratify errors for single-sample classes
        
        Args:
            X: Images array
            y: Labels array
            test_size: Fraction for validation set
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        logger = logging.getLogger(__name__)
        
        # Check if any class has only 1 sample
        unique_classes, counts = np.unique(y, return_counts=True)
        min_count = np.min(counts)
        classes_with_one = unique_classes[counts == 1]
        
        if min_count < 2:
            logger.warning(f"{len(classes_with_one)} classes have only 1 sample. Using non-stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )
        else:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y, shuffle=True
                )
            except Exception as e:
                logger.warning(f"Stratified split failed ({e}). Using non-stratified split.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=42, shuffle=True
                )
        
        return X_train, X_val, y_train, y_val
    
    def load_training_data(self, augment=True):
        """
        Load and preprocess training data from directory structure
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            X_train, y_train, X_val, y_val
        """
        logger = logging.getLogger(__name__)
        train_dir = self.data_dir / "processed" / "train" / "primary_core_signs"
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        # Get class names from folder names
        self.class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(self.class_names)} classes")
        
        images = []
        labels = []
        class_counts = []
        
        # Load images from each class folder
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = train_dir / class_name
            image_files = (
                list(class_dir.glob("*.png")) + 
                list(class_dir.glob("*.jpg")) + 
                list(class_dir.glob("*.jpeg")) +
                list(class_dir.glob("*.bmp"))
            )
            
            n_images = len(image_files)
            class_counts.append((class_name, n_images))
            logger.info(f"  {class_name}: {n_images} image(s)")
            
            for image_file in image_files:
                try:
                    processed_image = self.normalizer.process_image(image_file)
                    images.append(processed_image)
                    labels.append(class_idx)
                except Exception as e:
                    logger.error(f"Error loading {image_file}: {e}")
        
        if len(images) == 0:
            raise RuntimeError("No training images were loaded. Check the dataset directory.")
        
        X = np.array(images)
        y = np.array(labels)
        
        # Reshape for CNN (add channel dimension)
        X = X.reshape(X.shape[0], 64, 64, 1)
        
        logger.info(f"Total raw training samples: {len(X)}")
        
        # Augment first, then split (so augmented versions stay with their originals in train/val)
        if augment:
            X, y = self._augment_dataset(X, y)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = self._safe_train_val_split(X, y, test_size=0.15)
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        
        # Log class distribution summary
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        val_unique, val_counts = np.unique(y_val, return_counts=True)
        logger.info(f"Train classes: {len(train_unique)}, Val classes: {len(val_unique)}")
        
        return X_train, y_train, X_val, y_val
    
    def build_model(self, num_classes=None):
        """
        Build the classifier model
        
        Args:
            num_classes: Number of classes (uses len(class_names) if None)
        """
        logger = logging.getLogger(__name__)
        
        if num_classes is None:
            num_classes = len(self.class_names)
        
        self.classifier = IndusClassifierCNN(
            input_shape=(64, 64, 1),
            num_classes=num_classes
        )
        self.classifier.build_model()
        logger.info("Model built successfully")
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=80, batch_size=16):
        """
        Train the classifier
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training for {epochs} epochs (batch_size={batch_size})...")
        
        # Adjust epochs for very small datasets
        if len(X_train) < 200:
            epochs = max(epochs, 60)
            logger.info(f"Small dataset detected. Using {epochs} epochs for adequate learning.")
        
        history = self.classifier.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Log final metrics
        if hasattr(history, 'history'):
            h = history.history
            if 'accuracy' in h and 'val_accuracy' in h:
                final_acc = h['accuracy'][-1]
                final_val_acc = h['val_accuracy'][-1]
                best_val_acc = max(h['val_accuracy'])
                logger.info(f"Final training accuracy: {final_acc:.4f}")
                logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
                logger.info(f"Best validation accuracy: {best_val_acc:.4f} (epoch {np.argmax(h['val_accuracy'])+1})")
        
        return history
    
    def save_trained_model(self, model_name="indus_classifier"):
        """
        Save the trained model
        
        Args:
            model_name: Name for the saved model
        """
        logger = logging.getLogger(__name__)
        
        model_path = self.model_dir / f"{model_name}.keras"
        self.classifier.save_model(str(model_path))
        
        # Save class names
        class_names_path = self.model_dir / f"{model_name}_classes.txt"
        with open(class_names_path, 'w') as f:
            f.write('\n'.join(self.class_names))
        
        logger.info(f"Model and class names saved to {self.model_dir}")
    
    def load_trained_model(self, model_name="indus_classifier"):
        """
        Load a previously trained model
        
        Args:
            model_name: Name of the saved model
        """
        logger = logging.getLogger(__name__)
        
        model_path = self.model_dir / f"{model_name}.keras"
        class_names_path = self.model_dir / f"{model_name}_classes.txt"
        
        self.classifier.load_model(str(model_path))
        
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Loaded {len(self.class_names)} classes")


def main():
    """Main training pipeline"""
    
    # Set paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    model_dir = project_root / "models"
    log_dir = data_dir / "results" / "logs"
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("=" * 60)
    logger.info("INDUS-KEELADI CNN TRAINING PIPELINE STARTED")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = IndusKeeladiTrainer(data_dir, model_dir, augment_factor=25)
    
    # Load training data with augmentation
    logger.info("Loading training data with augmentation...")
    try:
        X_train, y_train, X_val, y_val = trainer.load_training_data(augment=True)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise
    
    # Build model
    logger.info("Building CNN model...")
    trainer.build_model()
    
    # Display model summary
    trainer.classifier.get_model_summary()
    
    # Train model
    logger.info("Beginning model training...")
    history = trainer.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=80,
        batch_size=16
    )
    
    # Save trained model
    logger.info("Saving trained model...")
    trainer.save_trained_model("indus_classifier")
    
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
