"""
Streamlit Dashboard — Marine Diesel Engine Fault Diagnosis System
Lebanese American University — Thesis Project

Run: streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from data_preparation import FAULT_LABELS, FAULT_NAMES, FAULT_COLORS, FEATURES
from severity_estimation import predict_severity

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Diesel Engine Fault Diagnosis',
    page_icon='🔧',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1.5rem;
    }
    .fault-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        margin: 0.3rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 5px solid;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .stSlider > div > div > div > div { height: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    ann_path = os.path.join('results', 'models', 'ann_model.pkl')
    sev_path = os.path.join('results', 'models', 'severity_models.pkl')
    if not os.path.exists(ann_path) or not os.path.exists(sev_path):
        st.error('Models not found. Run: python main.py --model first.')
        st.stop()
    with open(ann_path, 'rb') as f:
        ann_bundle = pickle.load(f)
    with open(sev_path, 'rb') as f:
        sev_models = pickle.load(f)
    return ann_bundle, sev_models

ann_bundle, sev_models = load_models()
ann_model = ann_bundle['model']
scaler    = ann_bundle['scaler']

# ── Fault descriptions ────────────────────────────────────────────────────────
FAULT_DESC = {
    'HE': (
        '✅ Engine operating normally.',
        'All parameters within acceptable limits. No maintenance action required. '
        'Continue routine monitoring schedule.'
    ),
    'ITE': (
        '⚡ Injection Timing Early detected.',
        'Fuel injection occurs before top dead centre, causing early combustion and '
        'high peak pressure. Risk of bearing damage and engine knock. '
        'Action: retard injection timing immediately.'
    ),
    'ITL': (
        '🐢 Injection Timing Late detected.',
        'Combustion occurs during the expansion stroke, reducing power output and '
        'increasing exhaust temperature. Action: advance injection timing, '
        'check fuel pump timing.'
    ),
    'PRW': (
        '💧 Piston Ring Wear detected.',
        'Worn piston rings cause poor compression and blow-by, reducing power and '
        'increasing lubricant consumption. Action: schedule piston ring inspection '
        'and replacement during next overhaul.'
    ),
    'SAPD': (
        '🚧 Scavenge Air Port Deposits detected.',
        'Carbon and oil deposits blocking scavenge ports restrict air flow, '
        'causing poor combustion and rising exhaust temperature. '
        'Action: clean scavenge ports at earliest opportunity.'
    ),
}

# ── Preset Examples ───────────────────────────────────────────────────────────
PRESETS = {
    'Select a preset…': (0.0, 0.0, 0.0, 0.0, 0.0),
    'Healthy Engine':                      (-0.2, 0.2, 0.3, 0.2, -0.5),
    'ITE — 50% severity':                  (7.5, 7.5, 0.0, 4.3, -6.3),
    'ITE — 90% severity':                  (13.0, 13.0, 0.0, 6.5, -11.3),
    'ITL — 50% severity':                  (-10.5, -10.5, -0.2, -5.2, 6.8),
    'ITL — 90% severity':                  (-19.2, -19.2, -0.4, -11.0, 11.3),
    'PRW — 50% severity':                  (-5.1, -5.2, -4.2, -5.1, 5.2),
    'PRW — 90% severity':                  (-9.2, -10.8, -9.6, -9.2, 10.5),
    'SAPD — 50% severity':                 (-8.0, -8.0, -6.5, -5.0, 25.5),
    'SAPD — 90% severity':                 (-14.0, -14.0, -12.0, -9.5, 59.5),
}

# ── Confidence Chart ──────────────────────────────────────────────────────────
def make_confidence_chart(proba):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = [FAULT_COLORS[c] for c in FAULT_LABELS]
    y_pos  = range(len(FAULT_LABELS))
    bars   = ax.barh(list(y_pos), proba * 100, color=colors,
                     edgecolor='white', height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(FAULT_LABELS, fontsize=11)
    ax.set_xlabel('Confidence (%)', fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_title('Class Probabilities', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, proba):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=10, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


# ── Severity Gauge ────────────────────────────────────────────────────────────
def make_severity_gauge(severity, fault):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    color = FAULT_COLORS.get(fault, '#95a5a6')
    ax.barh([0], [severity], color=color, height=0.4,
            edgecolor='white', alpha=0.85)
    ax.barh([0], [90 - severity], left=[severity], color='#ecf0f1',
            height=0.4, edgecolor='white')
    ax.set_xlim(0, 90)
    ax.set_yticks([])
    ax.set_xlabel('Severity (%)', fontsize=10)
    ax.set_title(f'Estimated Severity: {severity:.0f}%',
                 fontsize=11, fontweight='bold', color=color)
    for x in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        ax.axvline(x, color='white', linewidth=0.8, alpha=0.8)
    ax.text(severity, 0, f'{severity:.0f}%', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


# ── Radar Input Visualisation ─────────────────────────────────────────────────
def make_input_radar(values):
    from math import pi
    feats  = ['MIP', 'IKW', 'PCOM', 'PMAX', 'EXH']
    ranges = [(-25, 15), (-25, 15), (-15, 5), (-15, 10), (-15, 65)]
    normed = [(v - lo) / (hi - lo) * 2 - 1
              for v, (lo, hi) in zip(values, ranges)]

    N = len(feats)
    angles = [n / N * 2 * pi for n in range(N)] + [0]
    normed_plot = normed + [normed[0]]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feats, size=9)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-0.5, 0, 0.5]); ax.set_yticklabels([])
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax.plot(angles, normed_plot, 'o-', linewidth=2, color='#2980b9')
    ax.fill(angles, normed_plot, alpha=0.2, color='#2980b9')
    ax.set_title('Input Parameters\n(Normalised)', size=9,
                 fontweight='bold', y=1.12)
    plt.tight_layout()
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;font-size:2rem;">🔧 Marine Diesel Engine Fault Diagnosis</h1>
        <p style="margin:0.3rem 0 0 0;color:#666;font-size:1.05rem;">
            Lebanese American University &nbsp;|&nbsp; ANN-Based Diagnostic System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header('⚙️ Engine Parameters')
        st.caption('Enter percentage deviation from healthy baseline readings.')

        # Preset loader
        preset = st.selectbox('Load preset example:', list(PRESETS.keys()))
        pv = PRESETS[preset]

        st.divider()

        mip  = st.slider('MIP Deviation (%)',  -25.0, 20.0,  float(pv[0]), 0.1)
        ikw  = st.slider('IKW Deviation (%)',  -25.0, 20.0,  float(pv[1]), 0.1)
        pcom = st.slider('PCOM Deviation (%)', -15.0,  5.0,  float(pv[2]), 0.05)
        pmx  = st.slider('PMAX Deviation (%)', -15.0, 10.0,  float(pv[3]), 0.1)
        exh  = st.slider('EXH Deviation (%)',  -15.0, 65.0,  float(pv[4]), 0.1)

        st.divider()
        diagnose = st.button('🔍 Diagnose Engine', type='primary',
                             use_container_width=True)

        st.markdown('---')
        st.caption(
            '**Feature guide:**\n'
            '- MIP: Mean Indicated Pressure\n'
            '- IKW: Indicated Power\n'
            '- PCOM: Compression Pressure\n'
            '- PMAX: Maximum Pressure\n'
            '- EXH: Exhaust Temperature'
        )

    # ── Main content ──────────────────────────────────────────────────────────
    if not diagnose and preset == 'Select a preset…':
        # Welcome screen
        st.info(
            '👈 **Set engine parameters in the sidebar and click "Diagnose Engine"**\n\n'
            'Or select a preset example to see the system in action.',
            icon='💡'
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('📊 System Overview')
            st.markdown("""
            This system identifies **4 fault types** in marine diesel engines using
            an Artificial Neural Network trained on experimental data (Table 3.1).

            | Fault | Full Name |
            |-------|-----------|
            | **HE**   | Healthy Engine |
            | **ITE**  | Injection Timing Early |
            | **ITL**  | Injection Timing Late |
            | **PRW**  | Piston Ring Wear |
            | **SAPD** | Scavenge Air Port Deposits |
            """)
        with col2:
            st.subheader('📈 Model Performance')
            st.markdown("""
            | Metric | Value |
            |--------|-------|
            | Test Accuracy | **99.2%** |
            | 5-Fold CV Accuracy | **99.4% ± 0.8%** |
            | Macro F1 Score | **0.9938** |
            | Macro AUC (ROC) | **~1.000** |
            | Architecture | **MLP: 5→25→15→5** |
            """)
        return

    # ── Run prediction ────────────────────────────────────────────────────────
    sample = np.array([[mip, ikw, pcom, pmx, exh]])
    sample_scaled = scaler.transform(sample)
    pred_idx = ann_model.predict(sample_scaled)[0]
    proba    = ann_model.predict_proba(sample_scaled)[0]
    fault    = FAULT_LABELS[pred_idx]
    confidence = proba[pred_idx]

    # Severity
    severity = predict_severity(sev_models, fault, [mip, ikw, pcom, pmx, exh])

    # ── Results layout ────────────────────────────────────────────────────────
    col_res, col_chart, col_radar = st.columns([2, 2, 1.5])

    with col_res:
        st.subheader('🔬 Diagnosis Result')
        color = FAULT_COLORS[fault]
        st.markdown(
            f'<div class="fault-badge" style="background:{color};">'
            f'{fault} — {FAULT_NAMES[fault]}</div>',
            unsafe_allow_html=True,
        )
        st.metric('Confidence',  f'{confidence:.1%}')
        sev_display = '—' if fault == 'HE' else f'{severity:.0f}%'
        st.metric('Estimated Severity', sev_display)

        # Fault description
        short_desc, long_desc = FAULT_DESC[fault]
        st.markdown(f'<div class="info-box"><b>{short_desc}</b><br>{long_desc}</div>',
                    unsafe_allow_html=True)

    with col_chart:
        st.subheader('📊 Class Probabilities')
        fig_conf = make_confidence_chart(proba)
        st.pyplot(fig_conf, use_container_width=True)
        plt.close()

        if fault != 'HE':
            fig_sev = make_severity_gauge(severity, fault)
            st.pyplot(fig_sev, use_container_width=True)
            plt.close()

    with col_radar:
        st.subheader('📡 Input Profile')
        fig_radar = make_input_radar([mip, ikw, pcom, pmx, exh])
        st.pyplot(fig_radar, use_container_width=True)
        plt.close()

    # ── Parameter summary table ───────────────────────────────────────────────
    st.divider()
    st.subheader('📋 Parameter Summary')
    feat_names = ['MIP Deviation (%)', 'IKW Deviation (%)',
                  'PCOM Deviation (%)', 'PMAX Deviation (%)',
                  'EXH Deviation (%)']
    values_in  = [mip, ikw, pcom, pmx, exh]
    df_params = {
        'Parameter': feat_names,
        'Input Value (%)': values_in,
        'Status': [
            '🟢 Normal' if abs(v) < 2 else
            '🟡 Caution' if abs(v) < 8 else '🔴 Alert'
            for v in values_in
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(df_params), use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
