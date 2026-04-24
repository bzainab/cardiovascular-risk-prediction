"""
Hyperparameter tuning and model optimization module.
Uses GridSearchCV and threshold optimization to improve model performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
import joblib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelOptimizer:
    """Optimizes models through hyperparameter tuning and threshold adjustment."""
    
    def __init__(self):
        self.best_models = {}
        self.tuning_results = {}
        self.threshold_analysis = {}
        
    def tune_logistic_regression(self, X_train, y_train):
        """Tune Logistic Regression hyperparameters."""
        logger.info("\nTuning Logistic Regression...")
        
        from sklearn.linear_model import LogisticRegression
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': ['balanced', None],
            'max_iter': [500, 1000, 2000],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        }
        
        lr = LogisticRegression(random_state=42)
        
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.best_models['Logistic Regression (Tuned)'] = grid_search.best_estimator_
        self.tuning_results['Logistic Regression'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        return grid_search.best_estimator_
    
    def tune_random_forest(self, X_train, y_train):
        """Tune Random Forest hyperparameters."""
        logger.info("\nTuning Random Forest...")
        
        from sklearn.ensemble import RandomForestClassifier
        
        param_grid = {
            'n_estimators': [200, 500, 800],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 8],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.best_models['Random Forest (Tuned)'] = grid_search.best_estimator_
        self.tuning_results['Random Forest'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        return grid_search.best_estimator_
    
    def tune_gradient_boosting(self, X_train, y_train):
        """Tune Gradient Boosting hyperparameters."""
        logger.info("\nTuning Gradient Boosting...")
        
        from sklearn.ensemble import GradientBoostingClassifier
        
        param_grid = {
            'n_estimators': [200, 400, 600],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [2, 3, 4, 5],
            'subsample': [0.7, 0.8, 0.9],
            'min_samples_split': [5, 10]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            gb, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.best_models['Gradient Boosting (Tuned)'] = grid_search.best_estimator_
        self.tuning_results['Gradient Boosting'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        return grid_search.best_estimator_
    
    def optimize_threshold(self, model, X_val, y_val, model_name):
        """Find optimal decision threshold based on F1 score."""
        logger.info(f"\nOptimizing threshold for {model_name}...")
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            from sklearn.metrics import precision_score, recall_score
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            
            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        logger.info(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        
        self.threshold_analysis[model_name] = {
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'thresholds': thresholds.tolist(),
            'f1_scores': f1_scores,
            'precisions': precisions,
            'recalls': recalls
        }
        
        return best_threshold
    
    def create_voting_ensemble(self, model_dict):
        """Create a voting ensemble from multiple models."""
        logger.info("\nCreating Voting Ensemble...")
        
        from sklearn.ensemble import VotingClassifier
        
        estimators = [(name, model) for name, model in model_dict.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability averaging
        )
        
        self.best_models['Voting Ensemble'] = voting_clf
        logger.info(f"Voting Ensemble created with {len(estimators)} models")
        
        return voting_clf
    
    def save_best_models(self, directory='../models'):
        """Save all tuned and ensemble models."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving optimized models to {directory}...")
        
        for name, model in self.best_models.items():
            save_path = directory / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_optimized.pkl"
            joblib.dump(model, save_path)
            logger.info(f"[OK] {name} saved to {save_path}")
    
    def get_optimization_summary(self):
        """Return summary of tuning results."""
        summary = {
            'tuned_models': list(self.best_models.keys()),
            'tuning_results': self.tuning_results,
            'threshold_analysis': self.threshold_analysis
        }
        return summary


def optimize_models_pipeline(X_train, y_train, X_val, y_val):
    """Main optimization pipeline."""
    logger.info("\n" + "="*70)
    logger.info("MODEL OPTIMIZATION & TUNING")
    logger.info("="*70)
    
    optimizer = ModelOptimizer()
    
    try:
        # Tune top performers
        optimizer.tune_logistic_regression(X_train, y_train)
        optimizer.tune_random_forest(X_train, y_train)
        optimizer.tune_gradient_boosting(X_train, y_train)
        
        # Optimize thresholds on validation set
        if 'Logistic Regression (Tuned)' in optimizer.best_models:
            optimizer.optimize_threshold(
                optimizer.best_models['Logistic Regression (Tuned)'],
                X_val, y_val, 'Logistic Regression (Tuned)'
            )
        
        # Create ensemble
        if len(optimizer.best_models) >= 2:
            ensemble_models = {
                name: model for name, model in optimizer.best_models.items()
                if 'Voting' not in name
            }
            optimizer.create_voting_ensemble(ensemble_models)
        
        # Save optimized models
        optimizer.save_best_models()
        
        logger.info(f"\n[OK] Optimization completed. Created {len(optimizer.best_models)} optimized models.")
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}", exc_info=True)
        raise
    
    return optimizer


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    
    data_path = Path(__file__).parent.parent / "Framingham Data" / "framingham.csv"
    preprocessor, X_train, X_test, y_train, y_test = preprocess_pipeline(str(data_path))
    
    # Split training into train/val for threshold optimization
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    optimizer = optimize_models_pipeline(X_tr, y_tr, X_val, y_val)
