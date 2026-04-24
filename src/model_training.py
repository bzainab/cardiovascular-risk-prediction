"""
Model training module for cardiovascular risk prediction.
Trains multiple ML models and compares stronger alternatives.
"""

from pathlib import Path
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelTrainer:
    """Trainer for multiple ML models."""
    
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        self.trained_models = {}
        
    def build_logistic_regression(self):
        """Build Logistic Regression model (balanced classes)."""
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        self.models['Logistic Regression (Balanced)'] = model
        return model

    def build_logistic_regression_accuracy(self):
        """Build Logistic Regression model tuned for raw accuracy."""
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=None
        )
        self.models['Logistic Regression (Accuracy)'] = model
        return model
    
    def build_random_forest(self):
        """Build Random Forest model."""
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1
        )
        self.models['Random Forest'] = model
        return model

    def build_extra_trees(self):
        """Build Extra Trees model."""
        model = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_split=8,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1
        )
        self.models['Extra Trees'] = model
        return model

    def build_gradient_boosting(self):
        """Build Gradient Boosting model."""
        model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42
        )
        self.models['Gradient Boosting'] = model
        return model

    def build_hist_gradient_boosting(self):
        """Build Histogram Gradient Boosting model."""
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42
        )
        self.models['Hist Gradient Boosting'] = model
        return model

    def build_svm(self):
        """Build SVM with RBF kernel."""
        model = SVC(
            C=2.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        self.models['SVM (RBF)'] = model
        return model
    
    def build_xgboost(self):
        """Build XGBoost model."""
        if xgb is None:
            logger.warning("XGBoost not installed. Skipping XGBoost model.")
            return None

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            scale_pos_weight=5.6,  # 84.8/15.2 ratio
            eval_metric='logloss',
            n_jobs=-1
        )
        self.models['XGBoost'] = model
        return model
    
    def build_ann(self):
        """Build Artificial Neural Network."""
        # MLPClassifier provides a lightweight ANN without requiring TensorFlow.
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )
        
        self.models['ANN'] = model
        return model
    
    def build_all_models(self, input_dim=None):
        """Build all models."""
        logger.info("Building all models...")
        self.build_logistic_regression()
        self.build_logistic_regression_accuracy()
        self.build_random_forest()
        self.build_extra_trees()
        self.build_gradient_boosting()
        self.build_hist_gradient_boosting()
        self.build_svm()
        self.build_xgboost()
        self.build_ann()
        logger.info(f"Built {len(self.models)} models: {list(self.models.keys())}")

    def train_model(self, model_name, X_train, y_train):
        """Train a single model by name."""
        logger.info(f"Training {model_name}...")
        model = self.models[model_name]

        if model_name == 'XGBoost':
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)

        self.trained_models[model_name] = model
        logger.info(f"[OK] {model_name} trained")
    
    def train_all_models(self, X_train, y_train):
        """Train all models."""
        logger.info("\n" + "="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)

        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
        
        logger.info(f"\n[OK] All {len(self.trained_models)} models trained successfully\n")
    
    def cross_validate(self, X_train, y_train, cv=5):
        """Perform stratified cross-validation for each model."""
        logger.info(f"\nPerforming {cv}-Fold Stratified Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            # ANN cross-validation is much slower; evaluate it on holdout set instead.
            if name == 'ANN':
                logger.info(f"{name}: Skipped in CV (evaluated on test set)")
                continue

            scores = cross_val_score(model, X_train, y_train, cv=skf, 
                                    scoring='roc_auc', n_jobs=-1)
            self.cv_scores[name] = scores
            logger.info(f"{name}: ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return self.cv_scores

    @staticmethod
    def _safe_model_filename(model_name):
        """Convert model display name into a filename-safe string."""
        safe = model_name.replace(' ', '_')
        safe = safe.replace('(', '').replace(')', '')
        return safe
    
    def save_models(self, directory='../models'):
        """Save all trained models."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving models to {directory}...")
        
        for name, model in self.trained_models.items():
            save_path = directory / f"{self._safe_model_filename(name)}.pkl"
            joblib.dump(model, save_path)
            
            logger.info(f"[OK] {name} saved to {save_path}")
    
    def get_models_summary(self):
        """Return summary of trained models."""
        return self.trained_models


def train_models_pipeline(X_train, y_train, X_test=None, y_test=None, models_dir=None):
    """Main training pipeline."""
    trainer = ModelTrainer()
    
    # Build all models
    trainer.build_all_models(input_dim=X_train.shape[1])
    
    # Cross-validation
    trainer.cross_validate(X_train, y_train, cv=3)
    
    # Train all models
    trainer.train_all_models(X_train, y_train)
    
    # Save models
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / 'models'
    trainer.save_models(directory=models_dir)
    
    return trainer


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    
    data_path = Path(__file__).parent.parent / "Framingham Data" / "framingham.csv"
    preprocessor, X_train, X_test, y_train, y_test = preprocess_pipeline(str(data_path))
    
    trainer = train_models_pipeline(X_train, y_train, X_test, y_test)
