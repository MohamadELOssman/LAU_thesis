"""
ANN Model — Marine Diesel Engine Fault Diagnosis.
Trains a Multi-Layer Perceptron and generates 6 publication-quality result figures.
"""
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize

from data_preparation import (
    prepare_datasets, FEATURES, FAULT_LABELS, FAULT_COLORS, FAULT_NAMES
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results', 'models')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
})

CLASS_COLORS = [FAULT_COLORS[c] for c in FAULT_LABELS]


# ── Model Definition ──────────────────────────────────────────────────────────
def build_model(random_state=42):
    """
    MLP: Input(5) → Dense(25, ReLU) → Dense(15, ReLU) → Output(5, Softmax-equiv).
    L2 regularisation (alpha=0.005), Adam optimiser, early stopping.
    """
    return MLPClassifier(
        hidden_layer_sizes=(25, 15),
        activation='relu',
        solver='adam',
        alpha=0.005,
        batch_size=32,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        tol=1e-5,
        random_state=random_state,
        verbose=False,
    )


# ── Figure 7: Loss & Validation Accuracy Curve ────────────────────────────────
def plot_loss_curve(model):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('ANN Training Convergence', fontsize=14, fontweight='bold')

    # Loss curve
    ax = axes[0]
    ax.plot(model.loss_curve_, color='#2980b9', linewidth=2, label='Training Loss')
    if hasattr(model, 'validation_scores_') and model.validation_scores_:
        # validation_scores_ are accuracy not loss; plot on twin axis
        ax2 = ax.twinx()
        ax2.plot(model.validation_scores_, color='#e74c3c', linewidth=2,
                 linestyle='--', label='Val Accuracy')
        ax2.set_ylabel('Validation Accuracy', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    else:
        ax.legend()

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (Cross-Entropy)', color='#2980b9')
    ax.tick_params(axis='y', labelcolor='#2980b9')
    ax.set_title('Loss & Validation Accuracy vs Epoch')
    ax.grid(True, alpha=0.4)
    ax.annotate(f'Converged at epoch {len(model.loss_curve_)}',
                xy=(len(model.loss_curve_)-1, model.loss_curve_[-1]),
                xytext=(len(model.loss_curve_)*0.6, model.loss_curve_[0]*0.7),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    # Learning curve (accuracy vs training size)
    ax = axes[1]
    ax.set_title('Learning Curve (Accuracy vs Training Size)')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Accuracy')
    ax.text(0.5, 0.5, '(See Figure 12 for full learning curve)',
            transform=ax.transAxes, ha='center', va='center',
            color='gray', fontsize=10)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '07_loss_curve.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 8: Confusion Matrix ────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'ANN Confusion Matrix   |   Overall Test Accuracy: {accuracy:.1%}',
                 fontsize=14, fontweight='bold')

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2%'],
        ['Count', 'Normalised (Row %)']
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            cmap='Blues', ax=ax,
            xticklabels=FAULT_LABELS,
            yticklabels=FAULT_LABELS,
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 12, 'weight': 'bold'}
        )
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontweight='bold')
        ax.set_ylabel('True Class', fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '08_confusion_matrix.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 9: Classification Report Heatmap ──────────────────────────────────
def plot_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred,
                                   target_names=FAULT_LABELS,
                                   output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    data = pd.DataFrame(
        {m: [report[c][m] for c in FAULT_LABELS] for m in metrics},
        index=FAULT_LABELS
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        data, annot=True, fmt='.3f', cmap='YlOrRd',
        vmin=0.7, vmax=1.0, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.8, 'label': 'Score'},
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    ax.set_title('Per-Class Classification Metrics  (Precision / Recall / F1)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Fault Class', fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)

    # Macro averages as text below
    macro = report['macro avg']
    fig.text(0.5, -0.04,
             f"Macro Avg — Precision: {macro['precision']:.3f}  "
             f"Recall: {macro['recall']:.3f}  "
             f"F1: {macro['f1-score']:.3f}",
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '09_classification_report.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 10: ROC Curves (One-vs-Rest) ──────────────────────────────────────
def plot_roc_curves(model, X_test, y_test):
    y_bin = label_binarize(y_test, classes=list(range(len(FAULT_LABELS))))
    y_score = model.predict_proba(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ROC Curves — One-vs-Rest Strategy', fontsize=14, fontweight='bold')

    # Individual class ROC
    ax = axes[0]
    ax.set_title('Per-Class ROC Curves')
    for i, (cls, color) in enumerate(zip(FAULT_LABELS, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{cls} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.4)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    # Micro/macro average ROC
    ax = axes[1]
    ax.set_title('Micro & Macro Average ROC')
    # Micro
    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    auc_m = auc(fpr_m, tpr_m)
    ax.plot(fpr_m, tpr_m, color='#2c3e50', linewidth=3,
            linestyle='-', label=f'Micro-avg (AUC = {auc_m:.3f})')
    # Macro
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_score[:, i])[0]
        for i in range(len(FAULT_LABELS))
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(FAULT_LABELS)):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_score[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= len(FAULT_LABELS)
    auc_macro = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color='#e74c3c', linewidth=3,
            linestyle='--', label=f'Macro-avg (AUC = {auc_macro:.3f})')
    ax.fill_between(all_fpr, mean_tpr, alpha=0.1, color='#e74c3c')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.4)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '10_roc_curves.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 11: Permutation Feature Importance ────────────────────────────────
def plot_feature_importance(model, X_test, y_test):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=1
    )
    importances = result.importances_mean
    stds = result.importances_std
    sorted_idx = np.argsort(importances)[::-1]

    feat_names = ['MIP Dev.%', 'IKW Dev.%', 'PCOM Dev.%', 'PMAX Dev.%', 'EXH Dev.%']
    colors_sorted = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(FEATURES))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range(len(FEATURES)),
        importances[sorted_idx],
        yerr=stds[sorted_idx],
        capsize=5,
        color=[CLASS_COLORS[i % 5] for i in range(len(FEATURES))],
        edgecolor='white', linewidth=1.2, width=0.55
    )
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([feat_names[i] for i in sorted_idx], fontsize=11)
    ax.set_ylabel('Mean Accuracy Decrease\n(Permutation Importance)')
    ax.set_title('Feature Importance — Permutation Method\n(Higher = More Important for Classification)',
                 fontweight='bold')
    ax.grid(True, axis='y', alpha=0.4)

    for bar, val in zip(bars, importances[sorted_idx]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '11_feature_importance.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 12: Learning Curve ──────────────────────────────────────────────────
def plot_learning_curve(model, X_train, y_train):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.15, 1.0, 10),
        scoring='accuracy',
        n_jobs=1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes, train_mean, 'o-', color='#2980b9', linewidth=2.5,
            markersize=7, label='Training Accuracy')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='#2980b9')
    ax.plot(train_sizes, val_mean, 's-', color='#e74c3c', linewidth=2.5,
            markersize=7, label='Cross-Validation Accuracy')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='#e74c3c')

    ax.axhline(val_mean[-1], color='gray', linestyle=':', linewidth=1.5,
               label=f'Final CV Accuracy = {val_mean[-1]:.1%}')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curve — ANN Fault Classifier\n(5-Fold Stratified Cross-Validation)',
                 fontweight='bold')
    ax.set_ylim([0.6, 1.05])
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '12_learning_curve.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Main Training & Evaluation ────────────────────────────────────────────────
def run_model():
    print('\n' + '='*55)
    print('  PHASE 3 — ANN MODEL TRAINING & EVALUATION')
    print('='*55)

    # ── Data
    X_train, X_test, y_train, y_test, scaler, label_map, df_raw, df_aug = \
        prepare_datasets(samples_per_class=120, test_size=0.20, noise_std=0.30)

    print(f'\n  Augmented dataset : {len(df_aug)} samples ({len(df_aug)//len(FAULT_LABELS)} per class)')
    print(f'  Train / Test      : {len(X_train)} / {len(X_test)} samples')
    print(f'  Features          : {FEATURES}')

    # ── Cross-Validation before final training
    cv_model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f'\n  5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    # ── Final model on full training set
    print('\n  Training final ANN model...')
    model = build_model()
    model.fit(X_train, y_train)
    print(f'  Training converged at epoch {len(model.loss_curve_)}')
    print(f'  Final training loss : {model.loss_curve_[-1]:.5f}')

    # ── Evaluation
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f'\n  Test Accuracy : {accuracy:.4f} ({accuracy:.1%})')
    print('\n  Classification Report:')
    print(classification_report(y_test, y_pred, target_names=FAULT_LABELS))

    # ── Save model
    model_path = os.path.join(MODELS_DIR, 'ann_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'label_map': label_map}, f)
    print(f'  Model saved: {model_path}')

    # ── Generate all result figures
    print('\n  Generating result figures...')
    plot_loss_curve(model)
    plot_confusion_matrix(y_test, y_pred, accuracy)
    plot_classification_report(y_test, y_pred)
    plot_roc_curves(model, X_test, y_test)
    plot_feature_importance(model, X_test, y_test)
    plot_learning_curve(model, X_train, y_train)

    print('\n  Model training complete — 6 figures saved.\n')
    return model, scaler, accuracy


if __name__ == '__main__':
    run_model()
