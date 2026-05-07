"""
Data preparation module for Marine Diesel Engine Fault Diagnosis.
Handles loading, augmentation, and preprocessing of Table 3.1 data.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

FEATURES = ['MIP_DIV', 'IKW_DIV', 'PCOM_DIV', 'PMX_DIV', 'EXH_DIV']
FAULT_LABELS = ['HE', 'ITE', 'ITL', 'PRW', 'SAPD']
FAULT_COLORS = {
    'HE':   '#27ae60',
    'ITE':  '#2980b9',
    'ITL':  '#e67e22',
    'PRW':  '#e74c3c',
    'SAPD': '#8e44ad',
}
FAULT_NAMES = {
    'HE':   'Healthy',
    'ITE':  'Injection Timing Early',
    'ITL':  'Injection Timing Late',
    'PRW':  'Piston Ring Wear',
    'SAPD': 'Scavenge Air Port Deposits',
}

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'engine_faults.csv')


def load_raw_data():
    """Load the original 40-row experimental dataset (Table 3.1)."""
    df = pd.read_csv(DATA_PATH)
    return df


def augment_data(df, samples_per_class=120, noise_std=0.30, random_state=42):
    """
    Generate a balanced augmented dataset by adding Gaussian noise.
    noise_std=0.30 represents realistic sensor measurement uncertainty (~0.3%).
    """
    rng = np.random.default_rng(random_state)
    balanced_frames = []

    for cls in FAULT_LABELS:
        cls_df = df[df['Fault_Type'] == cls][FEATURES + ['Severity', 'Fault_Type']].copy()
        n_orig = len(cls_df)
        n_needed = samples_per_class - n_orig

        orig_features = cls_df[FEATURES].values
        orig_meta = cls_df[['Severity', 'Fault_Type']].values

        aug_rows = []
        for i in range(n_needed):
            idx = i % n_orig
            noise = rng.normal(0, noise_std, len(FEATURES))
            new_features = orig_features[idx] + noise
            aug_rows.append(
                list(new_features) + list(orig_meta[idx])
            )

        aug_df = pd.DataFrame(aug_rows, columns=FEATURES + ['Severity', 'Fault_Type'])
        aug_df['Severity'] = aug_df['Severity'].astype(float)
        balanced_frames.append(pd.concat([cls_df, aug_df], ignore_index=True))

    return pd.concat(balanced_frames, ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)


def encode_labels(df):
    """Encode Fault_Type strings to integer class indices."""
    label_map = {cls: i for i, cls in enumerate(FAULT_LABELS)}
    return df['Fault_Type'].map(label_map).values, label_map


def prepare_datasets(samples_per_class=120, test_size=0.20, noise_std=0.30,
                     random_state=42):
    """
    Full pipeline: load → augment → encode → scale → split.
    Returns:
        X_train, X_test, y_train, y_test  (numpy arrays)
        scaler                              (fitted StandardScaler)
        label_map                           (dict str→int)
        df_raw                              (original 40-row DataFrame)
        df_aug                              (full augmented DataFrame)
    """
    df_raw = load_raw_data()
    df_aug = augment_data(df_raw, samples_per_class=samples_per_class,
                          noise_std=noise_std, random_state=random_state)

    X = df_aug[FEATURES].values
    y, label_map = encode_labels(df_aug)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, label_map, df_raw, df_aug
