"""
Custom Stacking Ensemble Model for Cardiovascular Risk Prediction
Combines Random Forest, Gradient Boosting, and XGBoost predictions
using a Logistic Regression meta-learner
"""

from pathlib import Path
import logging
import numpy as np
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StackingEnsembleTrainer:
    """Custom stacking ensemble combining RF, GB, and XGBoost with LR meta-learner."""
    
    def __init__(self, use_smote=True):
        self.stacking_model = None
        self.cv_scores = None
        self.optimal_threshold = None
        self.use_smote = use_smote
        self.meta_learner_weights = None
        
        logger.info(f"StackingEnsembleTrainer initialized (SMOTE: {use_smote})")
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance."""
        if not self.use_smote:
            return X_train, y_train
        
        logger.info("Applying SMOTE to training data...")
        logger.info(f"  Before - Positive: {(y_train == 1).sum()}, Negative: {(y_train == 0).sum()}")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"  After  - Positive: {(y_resampled == 1).sum()}, Negative: {(y_resampled == 0).sum()}")
        return X_resampled, y_resampled
    
    def find_optimal_threshold(self, y_true, y_proba):
        """Find optimal decision threshold using Youden's J statistic."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"  Optimal threshold: {optimal_threshold:.4f} (Youden's J: {j_scores[optimal_idx]:.4f})")
        return optimal_threshold
    
    def build_stacking_ensemble(self):
        """Build stacking classifier with RF, GB, XGBoost as base learners."""
        logger.info("\nBuilding Stacking Ensemble...")
        logger.info("="*60)
        
        # Define base learners
        base_learners = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                subsample=0.8
            )),
        ]
        
        # Add XGBoost if available
        if xgb is not None:
            scale_pos_weight = (4240 - 643) / 643
            base_learners.append(('xgb', xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )))
            logger.info("Base learners: Random Forest, Gradient Boosting, XGBoost")
        else:
            logger.warning("XGBoost not available, using RF and GB only")
            logger.info("Base learners: Random Forest, Gradient Boosting")
        
        # Define meta-learner
        meta_learner = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        )
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5  # 5-fold cross-validation for training meta-learner
        )
        
        logger.info("Meta-learner: Logistic Regression")
        logger.info("="*60)
        
        self.stacking_model = stacking_clf
        return stacking_clf
    
    def train_stacking_model(self, X_train, X_test, y_train, y_test):
        """Train the stacking ensemble."""
        logger.info("\n[STACKING MODEL] Training Started")
        logger.info("="*60)
        
        # Apply SMOTE
        X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)
        
        # Build stacking model
        self.build_stacking_ensemble()
        
        # Train on resampled data
        logger.info("Training stacking ensemble...")
        self.stacking_model.fit(X_train_resampled, y_train_resampled)
        logger.info("✓ Stacking ensemble trained")
        
        # Cross-validation score
        logger.info("Computing 5-fold cross-validation ROC-AUC...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.stacking_model, 
            X_train_resampled, 
            y_train_resampled,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        self.cv_scores = cv_scores
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Find optimal threshold on test set
        logger.info("Finding optimal decision threshold...")
        y_proba_test = self.stacking_model.predict_proba(X_test)[:, 1]
        self.optimal_threshold = self.find_optimal_threshold(y_test, y_proba_test)
        
        # Extract meta-learner weights (coefficients)
        meta_learner = self.stacking_model.final_estimator_
        self.meta_learner_weights = meta_learner.coef_[0]
        logger.info(f"Meta-learner weights: {self.meta_learner_weights}")
        
        logger.info("="*60)
        logger.info("[STACKING MODEL] Training Complete ✓")
        logger.info("="*60)
        
        return self.stacking_model
    
    def get_model_info(self):
        """Return model information as dictionary."""
        return {
            'model': self.stacking_model,
            'cv_scores': self.cv_scores,
            'optimal_threshold': self.optimal_threshold,
            'meta_learner_weights': self.meta_learner_weights
        }
    
    def save_model(self, output_dir='models'):
        """Save stacking model and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save model
        model_file = output_path / 'Stacking_Ensemble.pkl'
        joblib.dump(self.stacking_model, model_file)
        logger.info(f"Saved: {model_file}")
        
        # Save metadata
        metadata = {
            'cv_scores': self.cv_scores,
            'optimal_threshold': self.optimal_threshold,
            'meta_learner_weights': self.meta_learner_weights
        }
        metadata_file = output_path / 'stacking_metadata.pkl'
        joblib.dump(metadata, metadata_file)
        logger.info(f"Saved metadata: {metadata_file}")


def train_stacking_pipeline(X_train, X_test, y_train, y_test, models_output_dir='models'):
    """Main pipeline for stacking model training."""
    logger.info("\n" + "="*70)
    logger.info("STACKING ENSEMBLE TRAINING PIPELINE")
    logger.info("="*70)
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    logger.info(f"Class distribution - Positive: {(y_train == 1).sum()}, Negative: {(y_train == 0).sum()}")
    
    # Train stacking model
    trainer = StackingEnsembleTrainer(use_smote=True)
    model = trainer.train_stacking_model(X_train, X_test, y_train, y_test)
    
    # Save model
    trainer.save_model(output_dir=models_output_dir)
    
    return trainer, model
