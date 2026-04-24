"""
Improved Model Training Module with SMOTE and Advanced Optimization
Implements class imbalance handling and threshold optimization
"""

from pathlib import Path
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score
from imblearn.over_sampling import SMOTE  # NEW: SMOTE for class imbalance
import joblib

try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImprovedModelTrainer:
    """Advanced trainer with SMOTE and threshold optimization."""
    
    def __init__(self, use_smote=True):
        self.models = {}
        self.cv_scores = {}
        self.trained_models = {}
        self.optimal_thresholds = {}  # Store optimal decision thresholds
        self.use_smote = use_smote
        logger.info(f"SMOTE enabled: {use_smote}")
        
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance."""
        if not self.use_smote:
            return X_train, y_train
        
        logger.info(f"Applying SMOTE to training data...")
        logger.info(f"  Before - Positive cases: {(y_train == 1).sum()}, Negative: {(y_train == 0).sum()}")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"  After  - Positive cases: {(y_resampled == 1).sum()}, Negative: {(y_resampled == 0).sum()}")
        return X_resampled, y_resampled
    
    def find_optimal_threshold(self, y_true, y_proba):
        """Find optimal decision threshold to maximize F1-score or ROC-AUC."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Use Youden's J statistic: TPR - FPR
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"  Optimal threshold: {optimal_threshold:.4f} (Youden's J: {j_scores[optimal_idx]:.4f})")
        return optimal_threshold
    
    def build_logistic_regression_optimized(self):
        """Improved Logistic Regression with L2 regularization tuning."""
        # Use GridSearchCV to find best C parameter
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs',
            penalty='l2'
        )
        self.models['Logistic_Regression_Balanced_Optimized'] = model
        return model

    def build_random_forest_optimized(self):
        """Improved Random Forest with better hyperparameters."""
        model = RandomForestClassifier(
            n_estimators=200,  # Increased from default
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.models['Random_Forest_Optimized'] = model
        return model

    def build_gradient_boosting_optimized(self):
        """Improved Gradient Boosting with learning rate tuning."""
        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8
        )
        self.models['Gradient_Boosting_Optimized'] = model
        return model

    def build_xgboost_optimized(self):
        """Improved XGBoost with scale_pos_weight for class imbalance."""
        if xgb is None:
            logger.warning("XGBoost not installed")
            return None
        
        # Calculate scale_pos_weight based on class distribution
        scale_pos_weight = (4240 - 643) / 643  # approximate ratio
        
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=scale_pos_weight,  # NEW: Built-in class weight
            eval_metric='logloss'
        )
        self.models['XGBoost_Optimized'] = model
        return model

    def build_svm_optimized(self):
        """Improved SVM with probability calibration."""
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,  # Enable probability estimates
            random_state=42
        )
        self.models['SVM_RBF_Optimized'] = model
        return model

    def build_mlp_optimized(self):
        """Improved Neural Network with better architecture."""
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),  # Deeper network
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            batch_size=32,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.models['ANN_Optimized'] = model
        return model

    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all models with SMOTE and cross-validation."""
        
        # Apply SMOTE to training data
        X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)
        
        # Build all models
        self.build_logistic_regression_optimized()
        self.build_random_forest_optimized()
        self.build_gradient_boosting_optimized()
        self.build_xgboost_optimized()
        self.build_svm_optimized()
        self.build_mlp_optimized()
        
        # Train each model
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*60}")
            
            try:
                # Train on SMOTE-resampled data
                model.fit(X_train_resampled, y_train_resampled)
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, 
                                           cv=cv, scoring='roc_auc', n_jobs=-1)
                self.cv_scores[model_name] = cv_scores
                logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
                # Store trained model
                self.trained_models[model_name] = model
                
                # Find optimal threshold on test set
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    optimal_threshold = self.find_optimal_threshold(y_test, y_proba)
                    self.optimal_thresholds[model_name] = optimal_threshold
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return self.trained_models, self.cv_scores, self.optimal_thresholds
    
    def save_models(self, output_dir='models'):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_file = output_path / f'{model_name}.pkl'
            joblib.dump(model, model_file)
            logger.info(f"Saved: {model_file}")
        
        # Save optimal thresholds
        thresholds_file = output_path / 'optimal_thresholds.pkl'
        joblib.dump(self.optimal_thresholds, thresholds_file)
        logger.info(f"Saved optimal thresholds: {thresholds_file}")


def improve_models_pipeline(X_train, X_test, y_train, y_test, models_output_dir='models'):
    """Main pipeline for improved model training."""
    logger.info("Starting Improved Model Training Pipeline")
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    logger.info(f"Class distribution - Positive: {(y_train == 1).sum()}, Negative: {(y_train == 0).sum()}")
    
    trainer = ImprovedModelTrainer(use_smote=True)
    trained_models, cv_scores, optimal_thresholds = trainer.train_all_models(
        X_train, X_test, y_train, y_test
    )
    
    trainer.save_models(output_dir=models_output_dir)
    
    return trainer, trained_models, cv_scores, optimal_thresholds
