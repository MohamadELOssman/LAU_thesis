# Marine Diesel Engine Fault Diagnosis — ANN System
**Lebanese American University | Mechanical Engineering Thesis**

An end-to-end machine learning pipeline for diagnosing faults in marine diesel engines using an Artificial Neural Network. The system classifies four fault types from engine performance parameter deviations and estimates fault severity, supported by a live interactive dashboard.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Fault Types](#3-fault-types)
4. [Project Structure](#4-project-structure)
5. [Installation](#5-installation)
6. [Usage](#6-usage)
7. [Pipeline Results](#7-pipeline-results)
8. [Generated Figures](#8-generated-figures)
9. [Model Architecture](#9-model-architecture)
10. [Streamlit Dashboard](#10-streamlit-dashboard)

---

## 1. Project Overview

This project implements a two-stage fault diagnosis system for marine diesel engines:

- **Stage 1 — Fault Classification:** A Multi-Layer Perceptron (MLP) identifies whether the engine is healthy or suffering from one of four fault types.
- **Stage 2 — Severity Estimation:** A Random Forest regressor estimates the fault severity on a 0–90% scale.

The system is validated against 6 competing classifiers and presented via an interactive Streamlit dashboard.

---

## 2. Dataset

**Source:** Table 3.1 — *"The percentage deviation of each variable and its fault type and severity"* (experimental data from marine diesel engine trials).

**File:** `data/engine_faults.csv`

| Column | Description | Unit |
|--------|-------------|------|
| `MIP_DIV` | Mean Indicated Pressure deviation | % |
| `IKW_DIV` | Indicated Power deviation | % |
| `PCOM_DIV` | Compression Pressure deviation | % |
| `PMX_DIV` | Maximum Pressure deviation | % |
| `EXH_DIV` | Exhaust Outlet Temperature deviation | % |
| `Severity` | Fault severity level | % (0, 10, 20 … 90) |
| `Fault_Type` | Class label | HE / ITE / ITL / PRW / SAPD |

**Raw data:** 40 rows — 4 healthy baseline readings + 9 severity levels × 4 fault types.  
**Augmented training data:** 600 samples (120 per class) — Gaussian noise σ = 0.3% applied to simulate measurement uncertainty.

---

## 3. Fault Types

| Code | Full Name | Description |
|------|-----------|-------------|
| **HE** | Healthy Engine | All parameters within normal limits |
| **ITE** | Injection Timing Early | Fuel injects before TDC → high peak pressure, EXH decreases |
| **ITL** | Injection Timing Late | Combustion during expansion stroke → power loss, EXH increases |
| **PRW** | Piston Ring Wear | Poor compression and blow-by → PCOM drops, EXH rises |
| **SAPD** | Scavenge Air Port Deposits | Restricted air flow → PCOM drops, EXH rises sharply |

**Key discriminating patterns (from thesis analysis):**
- PCOM deviation is unaffected by ITE and ITL (≈ 0%)
- ITE effects on MIP, IKW, PMAX, EXH are the opposite of ITL
- EXH deviation increases for ITL, PRW, and SAPD — but at different rates
- SAPD produces the most extreme EXH deviation (up to +59.5% at 90% severity)

---

## 4. Project Structure

```
LAU_thesis/
├── data/
│   └── engine_faults.csv          # Raw dataset (Table 3.1)
│
├── src/
│   ├── data_preparation.py        # Load, augment, scale data
│   ├── eda.py                     # Exploratory data analysis (figures 1–6)
│   ├── ann_model.py               # ANN training & evaluation (figures 7–12)
│   ├── classifier_comparison.py   # 7-classifier benchmark (figures 13–16)
│   └── severity_estimation.py     # Severity regression (figures 17–18)
│
├── results/
│   ├── figures/                   # All 18 saved figures (PNG, 150 DPI)
│   └── models/
│       ├── ann_model.pkl          # Trained MLP + scaler
│       └── severity_models.pkl    # 4 RandomForest regressors (one per fault)
│
├── app.py                         # Streamlit interactive dashboard
├── main.py                        # CLI entry point for the full pipeline
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 5. Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/MohamadELOssman/LAU_thesis.git
cd LAU_thesis
```

### Step 2 — Create a Python virtual environment

A virtual environment keeps the project dependencies isolated from your system Python.

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Once activated, your terminal prompt will show `(venv)` — this means the environment is active and any packages you install stay inside the project folder.

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Deactivate when done (optional)
```bash
deactivate
```

> **Note:** Always activate the virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`) before running any project command in a new terminal session.

**Requirements:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `streamlit`

---

## 6. Usage

### Run the full pipeline (all phases)
```bash
python main.py
```
Executes EDA → ANN training → classifier comparison → severity estimation.  
Saves all 18 figures to `results/figures/` and models to `results/models/`.

### Run individual phases
```bash
python main.py --eda        # Phase 2 — EDA only
python main.py --model      # Phase 3 — Train ANN only
python main.py --compare    # Classifier comparison only
python main.py --severity   # Severity estimation only
```

### Predict a single sample (CLI)
```bash
python main.py --predict --mip -5.1 --ikw -5.2 --pcom -4.2 --pmx -5.1 --exh 5.2
```
Output example:
```
► Predicted Fault : PRW (Piston Ring Wear)
► Confidence      : 98.6%

  Class Probabilities:
    HE                                     0.1%
    ITE                                    0.0%
    ITL                                    0.1%
    PRW    █████████████████████████████   98.6%
    SAPD                                   1.3%
```

### Launch the interactive dashboard
```bash
streamlit run app.py
```

---

## 7. Pipeline Results

### Stage 1 — Fault Classification (ANN)

| Metric | Value |
|--------|-------|
| Test Accuracy | **99.2%** |
| 5-Fold CV Accuracy | **99.4% ± 0.8%** |
| Macro F1 Score | **0.9938** |
| Macro AUC (ROC) | **≈ 1.000** |
| Training convergence | epoch 89 |

**Per-class results:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| HE    | 0.96 | 1.00 | 0.98 |
| ITE   | 1.00 | 0.96 | 0.98 |
| ITL   | 1.00 | 1.00 | 1.00 |
| PRW   | 1.00 | 1.00 | 1.00 |
| SAPD  | 1.00 | 1.00 | 1.00 |

### Classifier Comparison (7 methods, 5-fold CV)

| Rank | Classifier | CV Accuracy | F1 |
|------|-----------|-------------|-----|
| 1 | K-Nearest Neighbors | 99.58% | 0.9959 |
| 2 | **ANN (Proposed)** | **99.38%** | **0.9938** |
| 2 | SVM | 99.38% | 0.9939 |
| 2 | Random Forest | 99.38% | 0.9938 |
| 5 | Decision Tree | 97.71% | 0.9768 |
| 6 | Logistic Regression | 93.75% | 0.9401 |
| 7 | Naive Bayes | 86.88% | 0.8702 |

> ANN is statistically tied with SVM and Random Forest. The 0.2% gap with KNN is within one standard deviation. ANN is preferred for real-time deployment (constant O(1) inference), extensibility to multi-fault conditions, and real-time streaming capability.

### Stage 2 — Severity Estimation (Random Forest Regressor)

| Fault | CV MAE (%) | CV R² |
|-------|-----------|-------|
| ITE   | 0.53 | 0.9983 |
| ITL   | 0.46 | 0.9987 |
| PRW   | 0.62 | 0.9968 |
| SAPD  | 0.44 | 0.9990 |

Severity is predicted within **< 1% error** across all fault types.

---

## 8. Generated Figures

All figures are saved to `results/figures/`.

### Phase 2 — Exploratory Data Analysis

| Figure | File | Description |
|--------|------|-------------|
| 01 | `01_fault_severity_profiles.png` | Line plots of all 5 parameters vs fault severity for each fault type |
| 02 | `02_feature_distributions.png` | Box plots of each feature grouped by fault class |
| 03 | `03_correlation_heatmap.png` | Pearson correlation matrix of all features |
| 04 | `04_radar_chart.png` | Radar chart showing fault signatures at maximum severity |
| 05 | `05_pairplot.png` | Pairwise scatter matrix for key discriminating features |
| 06 | `06_class_distribution.png` | Bar chart of class counts in the raw dataset |

### Phase 3 — ANN Model Results

| Figure | File | Description |
|--------|------|-------------|
| 07 | `07_loss_curve.png` | Training loss and validation accuracy vs epoch |
| 08 | `08_confusion_matrix.png` | Count and normalised confusion matrices (test set) |
| 09 | `09_classification_report.png` | Heatmap of precision, recall, F1 per class |
| 10 | `10_roc_curves.png` | Per-class and macro/micro-average ROC curves with AUC |
| 11 | `11_feature_importance.png` | Permutation feature importance (EXH is most important) |
| 12 | `12_learning_curve.png` | Accuracy vs training set size (5-fold stratified CV) |

### Classifier Comparison

| Figure | File | Description |
|--------|------|-------------|
| 13 | `13_classifier_accuracy_comparison.png` | CV accuracy and F1 bar chart for all 7 classifiers |
| 14 | `14_classifier_metrics_radar.png` | Radar chart comparing all classifiers across 4 metrics |
| 15 | `15_all_confusion_matrices.png` | Grid of normalised confusion matrices for all 7 classifiers |
| 16 | `16_roc_comparison.png` | Macro-average ROC curve comparison for all classifiers |

### Severity Estimation

| Figure | File | Description |
|--------|------|-------------|
| 17 | `17_severity_predicted_vs_actual.png` | Predicted vs actual severity scatter (one per fault type) |
| 18 | `18_severity_error_analysis.png` | MAE/RMSE bars, R² scores, and residual box plots |

---

## 9. Model Architecture

### Stage 1 — Fault Classifier (MLP)

```
Input Layer      →  5 neurons  (MIP, IKW, PCOM, PMAX, EXH deviations)
Hidden Layer 1   →  25 neurons  (ReLU activation)
Hidden Layer 2   →  15 neurons  (ReLU activation)
Output Layer     →  5 neurons  (Softmax — HE, ITE, ITL, PRW, SAPD)

Optimiser   : Adam
Regulariser : L2 (α = 0.005)
Max epochs  : 2000 (early stopping, patience = 40)
Batch size  : 32
Scaler      : StandardScaler (zero mean, unit variance)
```

### Stage 2 — Severity Estimator (Random Forest)

```
One regressor per fault type (ITE, ITL, PRW, SAPD)

n_estimators : 300
max_depth    : 8
min_samples_leaf : 2
Input        : 5 features (same as Stage 1)
Output       : severity in [0, 90] %
```

---

## 10. Streamlit Dashboard

Launch the interactive web dashboard:

```bash
streamlit run app.py
```

**Features:**
- **Parameter sliders** — adjust all 5 engine parameter deviations in real time
- **Preset examples** — 10 built-in examples (healthy + all faults at 50% and 90% severity)
- **Fault class badge** — colour-coded prediction with full fault name
- **Confidence chart** — horizontal bar chart showing probability for each class
- **Severity gauge** — progress bar showing estimated fault severity
- **Input radar chart** — normalised visualisation of the 5 input parameters
- **Parameter status table** — Normal / Caution / Alert flag per parameter
- **Fault description** — plain-language explanation and recommended action

---

*Lebanese American University — Mechanical Engineering Department*
