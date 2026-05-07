"""
Exploratory Data Analysis for Marine Diesel Engine Fault Diagnosis.
Generates 6 publication-quality figures saved to results/figures/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from math import pi
import os

from data_preparation import (
    load_raw_data, FEATURES, FAULT_LABELS, FAULT_COLORS, FAULT_NAMES
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_LABELS = {
    'MIP_DIV':  'MIP Dev. (%)',
    'IKW_DIV':  'IKW Dev. (%)',
    'PCOM_DIV': 'PCOM Dev. (%)',
    'PMX_DIV':  'PMAX Dev. (%)',
    'EXH_DIV':  'EXH Dev. (%)',
}

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
})


# ── 1. Fault Severity Profiles ────────────────────────────────────────────────
def plot_fault_severity_profiles(df):
    fault_types = ['ITE', 'ITL', 'PRW', 'SAPD']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Engine Parameter Deviations vs Fault Severity\n(Marine Diesel Engine — Table 3.1)',
                 fontsize=15, fontweight='bold', y=1.01)

    feature_styles = {
        'MIP_DIV':  ('-o',  '#2c3e50', 'MIP'),
        'IKW_DIV':  ('-s',  '#2980b9', 'IKW'),
        'PCOM_DIV': ('-^',  '#27ae60', 'PCOM'),
        'PMX_DIV':  ('-D',  '#e67e22', 'PMAX'),
        'EXH_DIV':  ('-*',  '#e74c3c', 'EXH', 9),
    }

    for ax, fault in zip(axes.flat, fault_types):
        sub = df[df['Fault_Type'] == fault].sort_values('Severity')
        for feat, style_info in feature_styles.items():
            fmt, color, label = style_info[0], style_info[1], style_info[2]
            ms = style_info[3] if len(style_info) > 3 else 6
            ax.plot(sub['Severity'], sub[feat], fmt, color=color,
                    label=label, markersize=ms, linewidth=2)

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'{fault} — {FAULT_NAMES[fault]}',
                     color=FAULT_COLORS[fault], fontweight='bold')
        ax.set_xlabel('Fault Severity (%)')
        ax.set_ylabel('Parameter Deviation (%)')
        ax.set_xticks(range(0, 100, 10))
        ax.legend(loc='best', ncol=2, framealpha=0.85)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '01_fault_severity_profiles.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── 2. Feature Box Plots per Fault Class ──────────────────────────────────────
def plot_feature_distributions(df):
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    fig.suptitle('Distribution of Engine Parameters by Fault Class',
                 fontsize=15, fontweight='bold')

    palette = [FAULT_COLORS[c] for c in FAULT_LABELS]
    order = FAULT_LABELS

    for ax, feat in zip(axes, FEATURES):
        sns.boxplot(data=df, x='Fault_Type', y=feat, order=order,
                    palette=palette, ax=ax, linewidth=1.5,
                    flierprops=dict(marker='o', markersize=3))
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(FEATURE_LABELS[feat], fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Deviation (%)')
        ax.set_xticklabels(order, rotation=30, ha='right')
        ax.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '02_feature_distributions.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── 3. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    corr_data = df[FEATURES].copy()
    corr_data.columns = [FEATURE_LABELS[f] for f in FEATURES]
    corr = corr_data.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='RdYlGn',
        vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix\n(All Fault Classes Combined)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '03_correlation_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── 4. Radar Chart — Fault Signatures at Max Severity ─────────────────────────
def plot_radar_chart(df):
    fault_types = ['ITE', 'ITL', 'PRW', 'SAPD']
    max_sev = df.groupby('Fault_Type')[FEATURES].apply(
        lambda g: g.iloc[g.index.get_loc(g.index.max())] if len(g) > 0 else g.iloc[-1]
    )
    # Use mean of top-2 severity rows per fault for stability
    top2 = df.groupby('Fault_Type').apply(
        lambda g: g.nlargest(2, 'Severity')[FEATURES].mean()
    ).loc[fault_types]

    # Normalise to [-1, 1] range for radar
    col_abs_max = df[FEATURES].abs().max()
    normalised = top2.div(col_abs_max)

    categories = [FEATURE_LABELS[f] for f in FEATURES]
    N = len(categories)
    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(-1, 1)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticklabels(['-0.5', '0', '0.5', '1.0'])
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)

    for fault in fault_types:
        values = normalised.loc[fault].values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5,
                color=FAULT_COLORS[fault], label=FAULT_NAMES[fault])
        ax.fill(angles, values, alpha=0.10, color=FAULT_COLORS[fault])

    ax.set_title('Fault Signatures at Maximum Severity\n(Normalised Parameter Deviations)',
                 size=13, fontweight='bold', y=1.12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '04_radar_chart.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── 5. Scatter Matrix (Pairplot) ──────────────────────────────────────────────
def plot_pairplot(df):
    plot_df = df.copy()
    plot_df.columns = [FEATURE_LABELS.get(c, c) for c in plot_df.columns]
    feat_cols = [FEATURE_LABELS[f] for f in FEATURES]

    palette = {FAULT_NAMES[k]: v for k, v in FAULT_COLORS.items()}
    plot_df['Fault Class'] = plot_df['Fault_Type'].map(FAULT_NAMES)

    # Keep only 3 most discriminative features for readability
    key_feats = ['MIP Dev. (%)', 'PCOM Dev. (%)', 'EXH Dev. (%)']
    g = sns.PairGrid(plot_df, vars=key_feats, hue='Fault Class',
                     palette=palette, diag_sharey=False)
    g.map_diag(sns.kdeplot, fill=True, alpha=0.5, linewidth=1.5)
    g.map_upper(sns.scatterplot, alpha=0.65, s=35, edgecolors='none')
    g.map_lower(sns.kdeplot, levels=4, alpha=0.7)
    g.add_legend(title='Fault Class', bbox_to_anchor=(1.05, 0.5),
                 loc='center left', framealpha=0.9)
    g.figure.suptitle('Pairwise Feature Relationships by Fault Class\n(Key Discriminating Features)',
                       fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '05_pairplot.png')
    g.figure.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── 6. Class Distribution Bar ──────────────────────────────────────────────────
def plot_class_distribution(df):
    counts = df['Fault_Type'].value_counts().reindex(FAULT_LABELS)
    colors = [FAULT_COLORS[c] for c in FAULT_LABELS]
    labels = [f"{c}\n{FAULT_NAMES[c]}" for c in FAULT_LABELS]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='white',
                  linewidth=1.2, width=0.6)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_title('Dataset Class Distribution\n(Original Experimental Data — Table 3.1)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Readings')
    ax.set_ylim(0, max(counts.values) * 1.2)
    ax.grid(True, axis='y', alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, '06_class_distribution.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Main ───────────────────────────────────────────────────────────────────────
def run_eda():
    print('\n' + '='*55)
    print('  PHASE 2 — EXPLORATORY DATA ANALYSIS')
    print('='*55)
    df = load_raw_data()
    print(f'\n  Dataset: {len(df)} rows  |  {df["Fault_Type"].nunique()} classes')
    print(f'  Features: {FEATURES}')
    print(f'\n  Class counts:\n{df["Fault_Type"].value_counts().to_string()}')

    print('\n  Generating figures...')
    plot_fault_severity_profiles(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_radar_chart(df)
    plot_pairplot(df)
    plot_class_distribution(df)
    print('\n  EDA complete — 6 figures saved.\n')


if __name__ == '__main__':
    run_eda()
