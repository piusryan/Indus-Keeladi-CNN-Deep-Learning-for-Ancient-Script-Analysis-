"""
Quick End-to-End Pipeline Runner for Demo
Runs training with fewer epochs, then evaluates on Keeladi data
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import IndusKeeladiTrainer, setup_logging


def run_quick_pipeline():
    """Run a quick train + evaluate pipeline"""
    
    data_dir = PROJECT_ROOT / "data"
    model_dir = PROJECT_ROOT / "models"
    output_dir = model_dir / "evaluation_results"
    log_dir = data_dir / "results" / "logs"
    
    logger = setup_logging(log_dir)
    logger.info("=" * 60)
    logger.info("QUICK END-TO-END PIPELINE DEMO")
    logger.info("=" * 60)
    
    # ── Step 1: Training ──────────────────────────────────────────
    logger.info("\n[STEP 1] Training CNN classifier (20 epochs)...")
    
    trainer = IndusKeeladiTrainer(data_dir, model_dir, augment_factor=20)
    
    logger.info("Loading training data with augmentation...")
    X_train, y_train, X_val, y_val = trainer.load_training_data(augment=True)
    
    logger.info("Building model...")
    trainer.build_model()
    trainer.classifier.get_model_summary()
    
    logger.info("Training (20 epochs, early stopping enabled)...")
    history = trainer.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=16
    )
    
    logger.info("Saving model...")
    trainer.save_trained_model("indus_classifier")
    logger.info("Model saved successfully!")
    
    # ── Step 2: Evaluation ────────────────────────────────────────
    logger.info("\n[STEP 2] Running Keeladi evaluation...")
    
    from src.evaluate import KeeladiEvaluator
    
    model_path = model_dir / "indus_classifier.keras"
    evaluator = KeeladiEvaluator(model_path, data_dir)
    
    logger.info("Loading Keeladi validation sets...")
    validation_data = evaluator.load_keeladi_validation_set()
    
    if not validation_data:
        logger.warning("No validation images found - skipping evaluation")
        return
    
    logger.info("Predicting matches (threshold=50%)...")
    predictions = evaluator.predict_keeladi_matches(validation_data, threshold=0.5)
    
    logger.info("Analyzing civilization link...")
    analysis = evaluator.analyze_civilization_link(predictions)
    
    logger.info("Generating reports...")
    evaluator.generate_report(predictions, analysis, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Reports: {output_dir}")
    logger.info("")
    logger.info("RESULTS SUMMARY:")
    logger.info(f"  Keeladi images analyzed: {analysis['total_keeladi_images']}")
    logger.info(f"  Match rate: {analysis['match_rate']:.2%}")
    logger.info(f"  Mean confidence: {analysis['mean_confidence']:.4f}")
    logger.info(f"  Unique Indus signs: {len(analysis['most_common_indus_signs'])}")
    if analysis['direct_matches']:
        for k, v in analysis['direct_matches'].items():
            logger.info(f"  {k}: {v['match_rate']:.0%} ({v['matches']}/{v['images']})")


if __name__ == "__main__":
    run_quick_pipeline()
