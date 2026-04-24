"""
Model evaluation module for cardiovascular risk prediction.
Evaluates trained models using multiple metrics (accuracy, precision, recall, F1, ROC-AUC).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, 
                            confusion_matrix, classification_report, auc)
import joblib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelEvaluator:
    """Evaluates and compares trained models."""
    
    def __init__(self):
        self.results = {}
        self.predictions = {}
        self.probabilities = {}
        self.results_df = None
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model."""
        logger.info(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            # Convert decision scores to probabilities for ROC-AUC calculation.
            scores = model.decision_function(X_test)
            y_pred_proba = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_pred_proba = y_pred.astype(float)
        
        # Store predictions
        self.predictions[model_name] = y_pred
        self.probabilities[model_name] = y_pred_proba
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results[model_name] = metrics
        
        # Log metrics
        logger.info(f"{model_name} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, trained_models, X_test, y_test):
        """Evaluate all trained models."""
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION")
        logger.info("="*60)
        
        for name, model in trained_models.items():
            self.evaluate_model(model, X_test, y_test, name)
        
        return self.results
    
    def compare_models(self):
        """Compare all models and return summary."""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)

        self.results_df = pd.DataFrame(self.results).T
        results_by_auc = self.results_df.sort_values('ROC-AUC', ascending=False)
        results_by_accuracy = self.results_df.sort_values('Accuracy', ascending=False)

        logger.info("\nPerformance Ranking (by ROC-AUC):")
        logger.info(results_by_auc.to_string())

        logger.info("\nPerformance Ranking (by Accuracy):")
        logger.info(results_by_accuracy.to_string())

        best_auc_model = results_by_auc.index[0]
        best_accuracy_model = results_by_accuracy.index[0]
        logger.info(
            f"\n[OK] Best ROC-AUC model: {best_auc_model} "
            f"(ROC-AUC: {results_by_auc.loc[best_auc_model, 'ROC-AUC']:.4f})"
        )
        logger.info(
            f"[OK] Best Accuracy model: {best_accuracy_model} "
            f"(Accuracy: {results_by_accuracy.loc[best_accuracy_model, 'Accuracy']:.4f})"
        )

        return results_by_auc
    
    def plot_roc_curves(self, y_test, output_path='../results/roc_curves.png'):
        """Plot ROC curves for all models."""
        logger.info(f"\nPlotting ROC curves...")
        
        plt.figure(figsize=(10, 7))
        
        for model_name, y_pred_proba in self.probabilities.items():
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"[OK] ROC curves saved to {output_path}")
        plt.close()
    
    def plot_confusion_matrices(self, y_test, output_path='../results/confusion_matrices.png'):
        """Plot confusion matrices for all models."""
        logger.info(f"\nPlotting confusion matrices...")
        
        n_models = len(self.predictions)
        n_cols = 2
        n_rows = int(np.ceil(n_models / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        axes = np.array(axes).reshape(-1)
        
        for idx, (model_name, y_pred) in enumerate(self.predictions.items()):
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                       cbar=False, square=True)
            axes[idx].set_title(f'{model_name}\nAccuracy: {self.results[model_name]["Accuracy"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"[OK] Confusion matrices saved to {output_path}")
        plt.close()
    
    def save_results(self, output_path='../results/model_results.csv'):
        """Save evaluation results to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(output_path, index_label='Model')
        logger.info(f"[OK] Results saved to {output_path}")
    
    def generate_report(self, y_test, best_model_name):
        """Generate detailed classification report for best model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"DETAILED REPORT - {best_model_name}")
        logger.info(f"{'='*60}")
        
        y_pred = self.predictions[best_model_name]
        report = classification_report(y_test, y_pred, 
                                      target_names=['No CHD', 'CHD'])
        logger.info(f"\n{report}")


def evaluate_models_pipeline(trained_models, X_test, y_test, results_dir='../results'):
    """Main evaluation pipeline."""
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    evaluator.evaluate_all_models(trained_models, X_test, y_test)
    
    # Compare and rank
    results_df = evaluator.compare_models()
    
    # Visualizations
    evaluator.plot_roc_curves(y_test, output_path=f'{results_dir}/roc_curves.png')
    evaluator.plot_confusion_matrices(y_test, output_path=f'{results_dir}/confusion_matrices.png')
    
    # Save results
    evaluator.save_results(output_path=f'{results_dir}/model_results.csv')
    
    # Best model report
    best_model = results_df.index[0]
    evaluator.generate_report(y_test, best_model)
    
    return evaluator, best_model


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    from model_training import train_models_pipeline
    
    data_path = Path(__file__).parent.parent / "Framingham Data" / "framingham.csv"
    preprocessor, X_train, X_test, y_train, y_test = preprocess_pipeline(str(data_path))
    
    trainer = train_models_pipeline(X_train, y_train, X_test, y_test)
    evaluator, best_model = evaluate_models_pipeline(trainer.trained_models, X_test, y_test)
