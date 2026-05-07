"""
Severity Estimation for Marine Diesel Engine Fault Diagnosis.
Stage 2 of the diagnostic pipeline: given fault type, estimate severity (0–90%).
Trains one Random Forest regressor per fault class on augmented data.
Generates 2 figures (17–18).
"""
import numpy as np
import pandas as pd
import pickle
import os, sys, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

sys.path.insert(0, os.path.dirname(__file__))
from data_preparation import (
    load_raw_data, augment_data,
    FEATURES, FAULT_LABELS, FAULT_COLORS, FAULT_NAMES
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results', 'models')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
    'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 10,
})

FAULT_TYPES = ['ITE', 'ITL', 'PRW', 'SAPD']


# ── Data Preparation ──────────────────────────────────────────────────────────
def build_severity_dataset(samples_per_class=120, noise_std=0.30, random_state=42):
    """
    Returns:
        train_data : dict fault → (X_train, y_train)  — augmented-ONLY (no originals)
        test_data  : dict fault → (X_test,  y_test)   — original measurements only

    augment_data() returns pd.concat([orig_rows, aug_rows]), so the first
    n_orig rows of each class are the original measurements. We strip them
    out of the training set so evaluation on originals is genuinely unseen.
    """
    df_raw = load_raw_data()
    df_aug = augment_data(df_raw, samples_per_class=samples_per_class,
                          noise_std=noise_std, random_state=random_state)

    train_data, test_data = {}, {}
    for fault in FAULT_TYPES:
        orig = df_raw[df_raw['Fault_Type'] == fault].sort_values('Severity')
        n_orig = len(orig)

        # Augmented block for this fault (all rows, sorted as concat returned)
        aug_block = df_aug[df_aug['Fault_Type'] == fault].reset_index(drop=True)
        # First n_orig rows are the originals — skip them
        aug_only = aug_block.iloc[n_orig:].reset_index(drop=True)

        test_data[fault] = (
            orig[FEATURES].values.astype(float),
            orig['Severity'].values.astype(float),
        )
        train_data[fault] = (
            aug_only[FEATURES].values.astype(float),
            aug_only['Severity'].values.astype(float),
        )
    return train_data, test_data


# ── Model ─────────────────────────────────────────────────────────────────────
def train_severity_models(train_data):
    """Train one RandomForestRegressor per fault type. Returns dict."""
    models = {}
    print(f'\n  {"Fault":<8}  {"Train N":>8}  {"CV MAE":>9}  {"CV R²":>8}')
    print('  ' + '-' * 40)
    for fault in FAULT_TYPES:
        X, y = train_data[fault]
        mdl = RandomForestRegressor(
            n_estimators=300, max_depth=8,
            min_samples_leaf=2, random_state=42
        )
        mdl.fit(X, y)
        cv_mae = -cross_val_score(mdl, X, y, cv=5,
                                  scoring='neg_mean_absolute_error').mean()
        cv_r2  = cross_val_score(mdl, X, y, cv=5, scoring='r2').mean()
        models[fault] = mdl
        print(f'  {fault:<8}  {len(X):>8}  {cv_mae:>9.2f}  {cv_r2:>8.4f}')
    return models


def evaluate_severity_models(models, test_data):
    """Evaluate on original (un-augmented) test readings. Returns results dict."""
    results = {}
    print(f'\n  {"Fault":<8}  {"MAE (%)":>9}  {"RMSE (%)":>10}  {"R²":>8}')
    print('  ' + '-' * 42)
    for fault in FAULT_TYPES:
        X, y_true = test_data[fault]
        y_pred = models[fault].predict(X)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        results[fault] = dict(y_true=y_true, y_pred=y_pred,
                              mae=mae, rmse=rmse, r2=r2)
        print(f'  {fault:<8}  {mae:>9.2f}  {rmse:>10.2f}  {r2:>8.4f}')
    return results


def predict_severity(models, fault_type, features):
    """Predict severity for a single sample. Returns float (0–90)."""
    if fault_type == 'HE':
        return 0.0
    if fault_type not in models:
        return float('nan')
    X = np.array(features).reshape(1, -1)
    return float(np.clip(models[fault_type].predict(X)[0], 0, 90))


# ── Figure 17: Predicted vs Actual Scatter ────────────────────────────────────
def plot_predicted_vs_actual(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Severity Estimation — Predicted vs Actual\n'
                 '(Random Forest Regressor, Evaluated on Original Measurements)',
                 fontsize=14, fontweight='bold')

    for ax, fault in zip(axes.flat, FAULT_TYPES):
        r = results[fault]
        y_true, y_pred = r['y_true'], r['y_pred']
        color = FAULT_COLORS[fault]

        ax.scatter(y_true, y_pred, color=color, s=90, zorder=5,
                   edgecolors='white', linewidth=0.8, label='Predictions')
        # Annotate each point with severity level
        for xt, xp in zip(y_true, y_pred):
            ax.annotate(f'{int(xt)}%', (xt, xp), textcoords='offset points',
                        xytext=(5, 3), fontsize=8, color='#555555')
        # Perfect prediction line
        lo, hi = 0, 100
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5,
                alpha=0.6, label='Perfect prediction')
        # Tolerance band ±10%
        ax.fill_between([lo, hi], [lo-10, hi-10], [lo+10, hi+10],
                        alpha=0.08, color=color, label='±10% band')

        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xlabel('Actual Severity (%)'); ax.set_ylabel('Predicted Severity (%)')
        ax.set_title(f'{fault} — {FAULT_NAMES[fault]}',
                     color=color, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.4)
        ax.text(0.97, 0.05,
                f"MAE  = {r['mae']:.2f}%\nRMSE = {r['rmse']:.2f}%\nR²   = {r['r2']:.4f}",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=color, alpha=0.9))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '17_severity_predicted_vs_actual.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {path}')


# ── Figure 18: Error Analysis Dashboard ──────────────────────────────────────
def plot_error_analysis(results):
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle('Severity Estimation — Error Analysis',
                 fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    faults  = FAULT_TYPES
    maes    = [results[f]['mae']  for f in faults]
    rmses   = [results[f]['rmse'] for f in faults]
    r2s     = [results[f]['r2']   for f in faults]
    colors  = [FAULT_COLORS[f]   for f in faults]

    # Left: MAE & RMSE grouped bars
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(faults))
    w = 0.35
    ax1.bar(x - w/2, maes,  w, label='MAE',  color=colors, alpha=0.85,
            edgecolor='white')
    ax1.bar(x + w/2, rmses, w, label='RMSE', color=colors, alpha=0.50,
            edgecolor='white', hatch='//')
    for xi, (m, r) in enumerate(zip(maes, rmses)):
        ax1.text(xi - w/2, m + 0.1, f'{m:.1f}', ha='center',
                 fontsize=9, fontweight='bold')
        ax1.text(xi + w/2, r + 0.1, f'{r:.1f}', ha='center',
                 fontsize=9, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(faults)
    ax1.set_ylabel('Error (%)'); ax1.set_title('MAE & RMSE per Fault Class')
    ax1.legend(); ax1.grid(True, axis='y', alpha=0.4)
    ax1.set_ylim(0, max(rmses) * 1.4)

    # Middle: R² bar chart
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(faults, r2s, color=colors, edgecolor='white', width=0.55)
    for bar, val in zip(bars, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylim(0.85, 1.01); ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score per Fault Class')
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, axis='y', alpha=0.4)

    # Right: Residuals box plot
    ax3 = fig.add_subplot(gs[2])
    residuals = [results[f]['y_pred'] - results[f]['y_true'] for f in faults]
    bp = ax3.boxplot(residuals, labels=faults, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    ax3.set_ylabel('Residual (Predicted − Actual) %')
    ax3.set_title('Residual Distribution per Fault Class')
    ax3.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '18_severity_error_analysis.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Main ───────────────────────────────────────────────────────────────────────
def run_severity_estimation():
    print('\n' + '='*55)
    print('  PHASE — SEVERITY ESTIMATION')
    print('='*55)

    train_data, test_data = build_severity_dataset()
    print(f'\n  Training one RandomForest regressor per fault type...')
    models = train_severity_models(train_data)

    print(f'\n  Evaluating on original experimental measurements...')
    results = evaluate_severity_models(models, test_data)

    # Save models
    sev_path = os.path.join(MODELS_DIR, 'severity_models.pkl')
    with open(sev_path, 'wb') as f:
        pickle.dump(models, f)
    print(f'\n  Models saved: {sev_path}')

    print('\n  Generating severity figures...')
    plot_predicted_vs_actual(results)
    plot_error_analysis(results)

    print('\n  Severity estimation complete — 2 figures saved (17–18).\n')
    return models, results


if __name__ == '__main__':
    run_severity_estimation()
