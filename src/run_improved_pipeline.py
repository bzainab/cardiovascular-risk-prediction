"""
Script to run improved model training with SMOTE
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import preprocess_pipeline
from improved_model_training import improve_models_pipeline
from stacking_model import train_stacking_pipeline
from model_evaluation import ModelEvaluator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / 'results' / 'improved_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("IMPROVED MODEL TRAINING WITH SMOTE")
    logger.info("="*70)
    
    # Phase 1: Data Preprocessing
    logger.info("\n[PHASE 1] DATA PREPROCESSING")
    logger.info("-"*70)
    
    data_path = Path(__file__).parent.parent / 'Framingham Data' / 'framingham.csv'
    preprocessor, X_train, X_test, y_train, y_test = preprocess_pipeline(
        data_path=data_path,
        test_size=0.2,
        random_state=42,
        preprocessor_output_path=str(Path(__file__).parent.parent / 'models' / 'preprocessor.pkl')
    )
    
    # Phase 2: Improved Model Training with SMOTE
    logger.info("\n[PHASE 2] IMPROVED MODEL TRAINING (WITH SMOTE)")
    logger.info("-"*70)
    
    trainer, trained_models, cv_scores, optimal_thresholds = improve_models_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models_output_dir=str(Path(__file__).parent.parent / 'models')
    )
    
    # Phase 2.5: Stacking Ensemble Training
    logger.info("\n[PHASE 2.5] STACKING ENSEMBLE TRAINING (CUSTOM MODEL)")
    logger.info("-"*70)
    
    stacking_trainer, stacking_model = train_stacking_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models_output_dir=str(Path(__file__).parent.parent / 'models')
    )
    
    # Add stacking model to trained_models for evaluation
    trained_models['Stacking_Ensemble'] = stacking_model
    
    # Phase 3: Model Evaluation
    logger.info("\n[PHASE 3] MODEL EVALUATION")
    logger.info("-"*70)
    
    evaluator = ModelEvaluator()
    results_dict = evaluator.evaluate_all_models(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test
    )
    
    # Compare models and create dataframe
    evaluator.compare_models()
    results_df = evaluator.results_df
    
    # Save results
    results_df.to_csv(Path(__file__).parent.parent / 'results' / 'improved_model_results.csv', index=False)
    logger.info(f"\n✅ Results saved to: results/improved_model_results.csv")
    
    logger.info("\n" + "="*70)
    logger.info("IMPROVED TRAINING COMPLETE")
    logger.info("="*70)
    
    # Print summary
    logger.info("\n📊 IMPROVED MODEL PERFORMANCE SUMMARY:\n")
    logger.info(results_df.to_string())
    
    return trainer, results_df

if __name__ == '__main__':
    trainer, results = main()
