"""
Main pipeline orchestrator for the complete ML workflow.
Runs preprocessing -> training -> evaluation in sequence.
"""

import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess_pipeline
from model_training import train_models_pipeline
from model_optimization import optimize_models_pipeline
from model_evaluation import evaluate_models_pipeline

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'pipeline.log'),
        logging.StreamHandler()
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete ML pipeline."""
    logger.info("\n" + "="*70)
    logger.info("CARDIOVASCULAR RISK PREDICTION - COMPLETE PIPELINE")
    logger.info("="*70 + "\n")
    
    # Paths
    project_root = PROJECT_ROOT
    data_path = project_root / "Framingham Data" / "framingham.csv"
    results_dir = project_root / "results"
    models_dir = project_root / "models"
    
    # Ensure directories exist
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Phase 1: Preprocessing
        logger.info("PHASE 1: DATA PREPROCESSING")
        logger.info("-" * 70)
        preprocessor, X_train, X_test, y_train, y_test = preprocess_pipeline(
            str(data_path),
            test_size=0.2,
            random_state=42,
            preprocessor_output_path=models_dir / 'preprocessor.pkl',
        )
        
        # Phase 2: Model Training
        logger.info("\n\nPHASE 2: MODEL TRAINING")
        logger.info("-" * 70)
        trainer = train_models_pipeline(
            X_train,
            y_train,
            X_test,
            y_test,
            models_dir=models_dir,
        )
        
        # Phase 3: Model Optimization (Hyperparameter Tuning)
        logger.info("\n\nPHASE 3: MODEL OPTIMIZATION")
        logger.info("-" * 70)
        # Split training set into train/validation for tuning and threshold optimization
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        optimizer = optimize_models_pipeline(X_tr, y_tr, X_val, y_val)
        
        # Phase 4: Model Evaluation
        logger.info("\n\nPHASE 4: MODEL EVALUATION")
        logger.info("-" * 70)
        
        # Combine original and optimized models for final evaluation
        combined_models = {**trainer.trained_models, **optimizer.best_models}
        
        evaluator, best_model = evaluate_models_pipeline(
            combined_models, X_test, y_test, 
            results_dir=str(results_dir)
        )
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"\n[OK] Best Model: {best_model}")
        logger.info(f"[OK] Total models trained: {len(combined_models)}")
        logger.info(f"[OK] Models saved to: {models_dir}")
        logger.info(f"[OK] Results saved to: {results_dir}")
        logger.info(f"[OK] Log file: {results_dir / 'pipeline.log'}")
        logger.info("\nNow ready for Web App integration!\n")
        
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
