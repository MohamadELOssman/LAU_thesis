"""
Marine Diesel Engine Fault Diagnosis — ANN System
Lebanese American University — Thesis Project

Usage:
    python main.py              # Run full pipeline
    python main.py --eda        # EDA only
    python main.py --model      # Train ANN only
    python main.py --compare    # Classifier comparison only
    python main.py --severity   # Severity estimation only
    python main.py --predict --mip -5.1 --ikw -5.2 --pcom -4.2 --pmx -5.1 --exh 5.2
    streamlit run app.py        # Launch interactive dashboard
"""
import sys
import os
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preparation import FAULT_LABELS, FAULT_NAMES, FEATURES
from eda import run_eda
from ann_model import run_model
from classifier_comparison import run_comparison
from severity_estimation import run_severity_estimation


def predict_fault(args):
    """Single-sample fault prediction from command-line inputs."""
    model_path = os.path.join('results', 'models', 'ann_model.pkl')
    if not os.path.exists(model_path):
        print('ERROR: Model not found. Run training first: python main.py --model')
        sys.exit(1)

    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)

    model  = bundle['model']
    scaler = bundle['scaler']

    sample = np.array([[args.mip, args.ikw, args.pcom, args.pmx, args.exh]])
    sample_scaled = scaler.transform(sample)
    pred_class = model.predict(sample_scaled)[0]
    proba = model.predict_proba(sample_scaled)[0]

    print('\n' + '='*50)
    print('  FAULT PREDICTION RESULT')
    print('='*50)
    print(f'\n  Input: MIP={args.mip}%  IKW={args.ikw}%  PCOM={args.pcom}%'
          f'  PMAX={args.pmx}%  EXH={args.exh}%')
    print(f'\n  ► Predicted Fault : {FAULT_LABELS[pred_class]}'
          f' ({FAULT_NAMES[FAULT_LABELS[pred_class]]})')
    print(f'  ► Confidence      : {proba[pred_class]:.1%}')
    print('\n  Class Probabilities:')
    for cls, p in zip(FAULT_LABELS, proba):
        bar = '█' * int(p * 30)
        print(f'    {cls:5s}  {bar:<30s}  {p:.1%}')
    print()


def print_banner():
    print('\n' + '='*60)
    print('  MARINE DIESEL ENGINE FAULT DIAGNOSIS — ANN SYSTEM')
    print('  Lebanese American University — Thesis Project')
    print('='*60)
    print('\n  Fault Classes: HE | ITE | ITL | PRW | SAPD')
    print('  Features:      MIP | IKW | PCOM | PMAX | EXH (% deviation)')
    print('  Model:         Multi-Layer Perceptron (25→15→5)')
    print()


def main():
    parser = argparse.ArgumentParser(description='Diesel Engine Fault Diagnosis ANN')
    parser.add_argument('--eda',      action='store_true', help='Run EDA only')
    parser.add_argument('--model',    action='store_true', help='Train ANN only')
    parser.add_argument('--compare',  action='store_true', help='Classifier comparison only')
    parser.add_argument('--severity', action='store_true', help='Severity estimation only')
    parser.add_argument('--predict',  action='store_true', help='Predict single sample')
    parser.add_argument('--mip',  type=float, default=0.0)
    parser.add_argument('--ikw',  type=float, default=0.0)
    parser.add_argument('--pcom', type=float, default=0.0)
    parser.add_argument('--pmx',  type=float, default=0.0)
    parser.add_argument('--exh',  type=float, default=0.0)
    args = parser.parse_args()

    print_banner()

    if args.predict:
        predict_fault(args)
    elif args.eda:
        run_eda()
    elif args.model:
        run_model()
    elif args.compare:
        run_comparison()
    elif args.severity:
        run_severity_estimation()
    else:
        # Full pipeline
        run_eda()
        run_model()
        run_comparison()
        run_severity_estimation()
        print('\n' + '='*60)
        print('  PIPELINE COMPLETE')
        print('  Figures (1–18) saved to: results/figures/')
        print('  Models saved to:         results/models/')
        print('  Dashboard: streamlit run app.py')
        print('='*60 + '\n')


if __name__ == '__main__':
    main()
