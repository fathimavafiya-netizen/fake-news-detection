"""
Evaluation module: metrics computation and visualization.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

from src.utils import save_metrics


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive classification metrics.

    Returns:
        dict with all metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
    }

    if y_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        metrics['avg_precision'] = float(average_precision_score(y_true, y_proba))

    return metrics


def print_metrics(metrics, model_name='Model'):
    """Print metrics in a formatted table."""
    print(f"\n--- {model_name} Metrics ---")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (weighted):{metrics['precision']:.4f}")
    print(f"  Recall (weighted):  {metrics['recall']:.4f}")
    print(f"  F1 Score (weighted):{metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    if 'avg_precision' in metrics:
        print(f"  Avg Precision:      {metrics['avg_precision']:.4f}")


def plot_confusion_matrix(y_true, y_pred, save_path, model_name='Model'):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Fake', 'Real']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_proba, save_path, model_name='Model'):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2,
            label=f'{model_name} (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_precision_recall_curve(y_true, y_proba, save_path, model_name='Model'):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='#4CAF50', lw=2,
            label=f'{model_name} (AP = {ap:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to {save_path}")


def plot_feature_importance(feature_names, importances, save_path,
                            top_n=20, title='Feature Importance'):
    """Plot horizontal bar chart of top features."""
    # Sort by importance
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#f44336' if v < 0 else '#4CAF50' for v in top_importances]
    ax.barh(range(top_n), top_importances, color=colors, alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance saved to {save_path}")


def full_evaluation(y_true, y_pred, y_proba, model_dir, prefix='test',
                    model_name='Model'):
    """
    Run full evaluation: compute metrics, plot confusion matrix, ROC, PR curves.
    """
    # Compute metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    print_metrics(metrics, model_name)

    # Save metrics
    save_metrics({f'{prefix}_{k}': v for k, v in metrics.items()}, model_dir)

    # Classification report
    print(f"\nClassification Report ({prefix}):")
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

    # Plots
    eval_dir = os.path.join(model_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(eval_dir, f'{prefix}_confusion_matrix.png'),
        model_name
    )

    if y_proba is not None:
        plot_roc_curve(
            y_true, y_proba,
            os.path.join(eval_dir, f'{prefix}_roc_curve.png'),
            model_name
        )
        plot_precision_recall_curve(
            y_true, y_proba,
            os.path.join(eval_dir, f'{prefix}_pr_curve.png'),
            model_name
        )

    return metrics
