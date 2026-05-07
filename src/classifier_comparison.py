"""
Classifier Comparison for Marine Diesel Engine Fault Diagnosis.
Benchmarks 7 classifiers against the ANN to justify the model choice.
Generates 4 publication-quality figures (13–16).
"""
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from math import pi

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_preparation import (
    prepare_datasets, FEATURES, FAULT_LABELS, FAULT_COLORS, FAULT_NAMES
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
})

# ── Classifier Definitions ─────────────────────────────────────────────────────
CLASSIFIERS = {
    'Decision\nTree':    DecisionTreeClassifier(max_depth=8, random_state=42),
    'K-Nearest\nNeighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'Naive\nBayes':      GaussianNB(),
    'Logistic\nRegress.': LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    'SVM':               SVC(kernel='rbf', C=10, gamma='scale',
                             probability=True, random_state=42),
    'Random\nForest':    RandomForestClassifier(n_estimators=200, max_depth=10,
                                                random_state=42),
    'ANN\n(Proposed)':   MLPClassifier(hidden_layer_sizes=(25, 15),
                                       activation='relu', solver='adam',
                                       alpha=0.005, max_iter=500,
                                       early_stopping=False,
                                       random_state=42),
}

CLF_COLORS = [
    '#95a5a6',  # DT     — grey
    '#3498db',  # KNN    — blue
    '#1abc9c',  # NB     — teal
    '#e67e22',  # LR     — orange
    '#9b59b6',  # SVM    — purple
    '#2ecc71',  # RF     — green
    '#e74c3c',  # ANN    — red (highlight)
]


# ── Cross-Validation Benchmark ────────────────────────────────────────────────
def benchmark_classifiers(X_train, y_train, n_splits=5):
    """Run 5-fold stratified CV for every classifier. Returns results DataFrame."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []
    print(f'\n  {"Classifier":<22} {"CV Acc":>9} {"±":>4} {"Prec":>8} {"Recall":>8} {"F1":>8} {"Time(s)":>9}')
    print('  ' + '-'*75)

    for name, clf in CLASSIFIERS.items():
        t0 = time.time()
        scores = cross_validate(
            clf, X_train, y_train, cv=cv,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=False, n_jobs=1
        )
        elapsed = time.time() - t0
        label = name.replace('\n', ' ')
        rec = {
            'Classifier': label,
            'Accuracy':   scores['test_accuracy'].mean(),
            'Acc_std':    scores['test_accuracy'].std(),
            'Precision':  scores['test_precision_macro'].mean(),
            'Recall':     scores['test_recall_macro'].mean(),
            'F1':         scores['test_f1_macro'].mean(),
            'Time':       elapsed,
        }
        records.append(rec)
        print(f'  {label:<22} {rec["Accuracy"]:>8.4f} {rec["Acc_std"]:>5.4f}'
              f' {rec["Precision"]:>8.4f} {rec["Recall"]:>8.4f}'
              f' {rec["F1"]:>8.4f} {elapsed:>9.2f}')

    return pd.DataFrame(records)


# ── Figure 13: Accuracy & F1 Bar Chart ────────────────────────────────────────
def plot_accuracy_comparison(results_df):
    df = results_df.copy()
    names = [n.replace(' ', '\n') for n in df['Classifier']]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Classifier Performance Comparison — 5-Fold Cross-Validation\n'
                 'Marine Diesel Engine Fault Diagnosis',
                 fontsize=14, fontweight='bold')

    for ax, metric, ylabel, title in zip(
        axes,
        ['Accuracy', 'F1'],
        ['Accuracy', 'Macro F1-Score'],
        ['Cross-Validation Accuracy', 'Macro F1-Score']
    ):
        bars = ax.bar(names, df[metric], color=CLF_COLORS,
                      edgecolor='white', linewidth=1.2, width=0.6)

        # Error bars for accuracy only
        if metric == 'Accuracy':
            ax.errorbar(range(len(df)), df[metric], yerr=df['Acc_std'],
                        fmt='none', color='#2c3e50', capsize=5, linewidth=1.5)

        # Highlight ANN bar with bold edge
        bars[-1].set_edgecolor('#c0392b')
        bars[-1].set_linewidth(2.5)

        # Value labels
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold')

        ax.set_ylim(0.60, 1.06)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_xticklabels(names, fontsize=10)
        ax.axhline(0.9, color='gray', linestyle=':', linewidth=1, alpha=0.7,
                   label='90% threshold')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.4)
        ax.set_axisbelow(True)

        # Annotate best
        best_idx = df[metric].idxmax()
        ax.annotate('Best', xy=(best_idx, df[metric][best_idx] + 0.018),
                    ha='center', fontsize=9, color='#c0392b', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '13_classifier_accuracy_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {path}')


# ── Figure 14: Metrics Radar Chart ────────────────────────────────────────────
def plot_metrics_radar(results_df):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    N = len(metrics)
    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=12, fontweight='bold')
    ax.set_ylim(0.6, 1.02)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.70', '0.80', '0.90', '1.00'], size=8)
    ax.grid(True, alpha=0.4)

    for (_, row), color in zip(results_df.iterrows(), CLF_COLORS):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        lw = 3.5 if 'ANN' in row['Classifier'] else 1.5
        ls = '-'  if 'ANN' in row['Classifier'] else '--'
        alpha = 0.25 if 'ANN' in row['Classifier'] else 0.06
        label = row['Classifier']
        ax.plot(angles, vals, ls, linewidth=lw, color=color, label=label)
        ax.fill(angles, vals, alpha=alpha, color=color)

    ax.set_title('Classifier Metrics Radar\n(Accuracy · Precision · Recall · F1)',
                 size=13, fontweight='bold', y=1.12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.18),
              framealpha=0.9, fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '14_classifier_metrics_radar.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 15: All Confusion Matrices ────────────────────────────────────────
def plot_all_confusion_matrices(X_train, X_test, y_train, y_test):
    n_clf = len(CLASSIFIERS)
    n_cols = 4
    n_rows = (n_clf + n_cols - 1) // n_cols   # ceiling division → 2 rows

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()
    fig.suptitle('Confusion Matrices — All Classifiers (Test Set)',
                 fontsize=14, fontweight='bold')

    for ax, (name, clf), color in zip(axes, CLASSIFIERS.items(), CLF_COLORS):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()
        cm = confusion_matrix(y_test, y_pred, normalize='true')

        border_color = '#c0392b' if 'ANN' in name else '#bdc3c7'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5 if 'ANN' in name else 0.8)

        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=FAULT_LABELS, yticklabels=FAULT_LABELS,
                    linewidths=0.3, linecolor='white', ax=ax,
                    cbar=False, annot_kws={'size': 9})
        label = name.replace('\n', ' ')
        ax.set_title(f'{label}\nAcc: {acc:.1%}',
                     fontweight='bold',
                     color='#c0392b' if 'ANN' in name else '#2c3e50',
                     fontsize=11)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide any unused subplots
    for ax in axes[n_clf:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '15_all_confusion_matrices.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 16: Macro ROC Comparison ──────────────────────────────────────────
def plot_roc_comparison(X_train, X_test, y_train, y_test):
    y_bin = label_binarize(y_test, classes=list(range(len(FAULT_LABELS))))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title('Macro-Average ROC Curve — All Classifiers',
                 fontsize=13, fontweight='bold')

    for (name, clf), color in zip(CLASSIFIERS.items(), CLF_COLORS):
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([
            roc_curve(y_bin[:, i], y_score[:, i])[0]
            for i in range(len(FAULT_LABELS))
        ]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(FAULT_LABELS)):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_score[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= len(FAULT_LABELS)
        macro_auc = auc(all_fpr, mean_tpr)

        label = name.replace('\n', ' ')
        lw    = 3.5 if 'ANN' in name else 1.5
        ls    = '-'  if 'ANN' in name else '--'
        ax.plot(all_fpr, mean_tpr, ls, color=color, linewidth=lw,
                label=f'{label}  (AUC = {macro_auc:.4f})',
                zorder=10 if 'ANN' in name else 5)

    ax.plot([0, 1], [0, 1], 'k:', linewidth=1.2, label='Random Classifier', zorder=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.legend(loc='lower right', framealpha=0.92, fontsize=10)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '16_roc_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Summary Table ──────────────────────────────────────────────────────────────
def print_summary_table(results_df):
    print('\n' + '='*75)
    print('  CLASSIFIER COMPARISON SUMMARY (5-Fold CV)')
    print('='*75)
    df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    print(df[['Rank', 'Classifier', 'Accuracy', 'Acc_std',
              'Precision', 'Recall', 'F1', 'Time']].to_string(
        index=False,
        float_format=lambda x: f'{x:.4f}'
    ))
    ann_row = df[df['Classifier'].str.contains('ANN')].iloc[0]
    print(f'\n  ANN Rank: #{int(ann_row["Rank"])} out of {len(df)} classifiers')
    print(f'  ANN Accuracy: {ann_row["Accuracy"]:.4f}  |  F1: {ann_row["F1"]:.4f}')
    print('='*75)


# ── Main ───────────────────────────────────────────────────────────────────────
def run_comparison():
    print('\n' + '='*55)
    print('  CLASSIFIER COMPARISON')
    print('='*55)

    X_train, X_test, y_train, y_test, scaler, label_map, df_raw, df_aug = \
        prepare_datasets(samples_per_class=120, test_size=0.20, noise_std=0.30)

    print(f'\n  Running 5-fold CV for {len(CLASSIFIERS)} classifiers...')
    results_df = benchmark_classifiers(X_train, y_train)

    print_summary_table(results_df)

    print('\n  Generating comparison figures...')
    plot_accuracy_comparison(results_df)
    plot_metrics_radar(results_df)
    plot_all_confusion_matrices(X_train, X_test, y_train, y_test)
    plot_roc_comparison(X_train, X_test, y_train, y_test)

    print('\n  Comparison complete — 4 figures saved (13–16).\n')
    return results_df


if __name__ == '__main__':
    run_comparison()
