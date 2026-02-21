"""
Sepsis Early Detection - Hackathon Starter Code
================================================
This code provides a complete pipeline for the PhysioNet 2019 Challenge dataset.
Run this to get a working baseline in under 30 minutes.

Author: Healthcare Hackathon Starter Kit
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# SECTION 1: DATA LOADING
# ==============================================================================

def load_challenge_data(data_dir: str, max_patients: int = None) -> pd.DataFrame:
    """
    Load PhysioNet Challenge 2019 data from PSV files.
    
    Args:
        data_dir: Path to extracted training data (training_setA or training_setB)
        max_patients: Limit number of patients for quick testing
    
    Returns:
        DataFrame with all patient data combined
    """
    all_data = []
    files = sorted(Path(data_dir).glob("*.psv"))
    
    if max_patients:
        files = files[:max_patients]
    
    for i, filepath in enumerate(files):
        if i % 1000 == 0:
            print(f"Loading patient {i}/{len(files)}...")
        
        patient_id = filepath.stem
        df = pd.read_csv(filepath, sep='|')
        df['patient_id'] = patient_id
        df['hour'] = range(len(df))
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


# Column definitions from PhysioNet Challenge 2019
VITAL_SIGNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

LAB_VALUES = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]

DEMOGRAPHICS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

ALL_FEATURES = VITAL_SIGNS + LAB_VALUES + DEMOGRAPHICS


# ==============================================================================
# SECTION 2: PREPROCESSING
# ==============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw data.
    """
    df = df.copy()
    
    # Forward fill missing values within each patient
    df[VITAL_SIGNS + LAB_VALUES] = df.groupby('patient_id')[VITAL_SIGNS + LAB_VALUES].ffill()
    
    # Fill remaining NaN with population median
    for col in VITAL_SIGNS + LAB_VALUES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Create missing indicators for important features
    for col in ['Lactate', 'WBC', 'Creatinine', 'Bilirubin_total']:
        df[f'{col}_missing'] = df[col].isna().astype(int)
    
    return df


def create_features(df: pd.DataFrame, window_hours: list = [3, 6, 12]) -> pd.DataFrame:
    """
    Create time-series features from raw measurements.
    """
    feature_dfs = []
    
    for patient_id, group in df.groupby('patient_id'):
        group = group.sort_values('hour')
        features = {'patient_id': patient_id, 'hour': group['hour'].values}
        
        # Rolling statistics for vital signs
        for col in VITAL_SIGNS:
            for w in window_hours:
                rolling = group[col].rolling(w, min_periods=1)
                features[f'{col}_mean_{w}h'] = rolling.mean().values
                features[f'{col}_std_{w}h'] = rolling.std().fillna(0).values
                features[f'{col}_min_{w}h'] = rolling.min().values
                features[f'{col}_max_{w}h'] = rolling.max().values
        
        # Trend features (slope)
        for col in VITAL_SIGNS[:4]:  # HR, O2Sat, Temp, SBP
            values = group[col].values
            slopes = np.zeros(len(values))
            for i in range(1, len(values)):
                start = max(0, i - 3)
                if i - start > 0:
                    slopes[i] = (values[i] - values[start]) / (i - start)
            features[f'{col}_slope_3h'] = slopes
        
        # Clinical derived features
        features['shock_index'] = (group['HR'] / group['SBP'].replace(0, np.nan)).fillna(1).values
        features['pulse_pressure'] = (group['SBP'] - group['DBP']).values
        
        # Original values
        for col in ALL_FEATURES + ['SepsisLabel']:
            if col in group.columns:
                features[col] = group[col].values
        
        feature_dfs.append(pd.DataFrame(features))
    
    return pd.concat(feature_dfs, ignore_index=True)


# ==============================================================================
# SECTION 3: MODEL TRAINING
# ==============================================================================

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost classifier for sepsis prediction.
    """
    import xgboost as xgb
    
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def train_baseline_models(X_train, y_train):
    """
    Train multiple baseline models for comparison.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    models = {}
    
    # Logistic Regression
    models['logistic'] = LogisticRegression(max_iter=1000, class_weight='balanced')
    models['logistic'].fit(X_train, y_train)
    
    # Random Forest
    models['random_forest'] = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
    )
    models['random_forest'].fit(X_train, y_train)
    
    return models


# ==============================================================================
# SECTION 4: EVALUATION
# ==============================================================================

def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """
    Evaluate model performance with multiple metrics.
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, 
        recall_score, f1_score, confusion_matrix
    )
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'Model': model_name,
        'AUROC': roc_auc_score(y_test, y_prob),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    print(f"\n{'='*50}")
    print(f"Results for {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"{metric:12s}: {value:.4f}")
    
    return metrics


def calculate_qsofa(df: pd.DataFrame) -> pd.Series:
    """
    Calculate qSOFA score for comparison.
    qSOFA ≥ 2 indicates high sepsis risk.
    """
    score = (
        (df['Resp'] >= 22).astype(int) +
        (df['SBP'] <= 100).astype(int) +
        (df['HR'] >= 90).astype(int)  # Proxy for altered mentation
    )
    return (score >= 2).astype(int)


# ==============================================================================
# SECTION 5: INTERPRETABILITY
# ==============================================================================

def explain_predictions(model, X_test, feature_names: list, num_samples: int = 100):
    """
    Generate SHAP explanations for model predictions.
    """
    import shap
    
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample
    sample_idx = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    X_sample = X_test[sample_idx] if isinstance(X_test, np.ndarray) else X_test.iloc[sample_idx]
    
    shap_values = explainer.shap_values(X_sample)
    
    return explainer, shap_values, X_sample


def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    """
    Plot feature importance from the trained model.
    """
    import matplotlib.pyplot as plt
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top Features for Sepsis Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("Feature importance plot saved to 'feature_importance.png'")


# ==============================================================================
# SECTION 6: MAIN PIPELINE
# ==============================================================================

def main():
    """
    Complete pipeline from data loading to evaluation.
    """
    print("="*60)
    print("SEPSIS EARLY DETECTION - HACKATHON PIPELINE")
    print("="*60)
    
    # Step 1: Check for data
    data_dir = "./training_setA"  # Update this path!
    
    if not os.path.exists(data_dir):
        print(f"""
        Data not found at {data_dir}
        
        Please download the PhysioNet Challenge 2019 data:
        1. Visit: https://physionet.org/content/challenge-2019/1.0.0/
        2. Download training_setA.zip and training_setB.zip
        3. Extract to the current directory
        4. Update data_dir path above
        
        For quick testing, generating synthetic data instead...
        """)
        df = generate_synthetic_data(n_patients=500)
    else:
        print("\n[1/6] Loading data...")
        df = load_challenge_data(data_dir, max_patients=2000)  # Limit for speed
    
    print(f"Loaded {df['patient_id'].nunique()} patients, {len(df)} total records")
    
    # Step 2: Preprocess
    print("\n[2/6] Preprocessing...")
    df = preprocess_data(df)
    
    # Step 3: Feature engineering
    print("\n[3/6] Creating features...")
    df_features = create_features(df)
    
    # Step 4: Prepare train/test split
    print("\n[4/6] Splitting data...")
    from sklearn.model_selection import train_test_split
    
    # Get feature columns
    feature_cols = [c for c in df_features.columns 
                    if c not in ['patient_id', 'hour', 'SepsisLabel']]
    
    X = df_features[feature_cols].values
    y = df_features['SepsisLabel'].values
    
    # Split by patient to avoid data leakage
    patients = df_features['patient_id'].unique()
    train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
    
    train_mask = df_features['patient_id'].isin(train_patients)
    test_mask = df_features['patient_id'].isin(test_patients)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Further split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Sepsis rate: {y_train.mean():.2%} (train), {y_test.mean():.2%} (test)")
    
    # Step 5: Train models
    print("\n[5/6] Training models...")
    
    # XGBoost (main model)
    print("  Training XGBoost...")
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Baselines
    print("  Training baselines...")
    baseline_models = train_baseline_models(X_train, y_train)
    
    # Step 6: Evaluate
    print("\n[6/6] Evaluating models...")
    
    results = []
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost"))
    
    for name, model in baseline_models.items():
        results.append(evaluate_model(model, X_test, y_test, name.replace('_', ' ').title()))
    
    # qSOFA baseline
    df_test = df_features[test_mask].copy()
    qsofa_pred = calculate_qsofa(df_test)
    qsofa_auroc = roc_auc_score(y_test, qsofa_pred) if y_test.sum() > 0 else 0.5
    print(f"\n{'='*50}")
    print(f"Results for qSOFA Baseline")
    print(f"{'='*50}")
    print(f"AUROC       : {qsofa_auroc:.4f}")
    
    # Feature importance
    print("\n" + "="*60)
    print("Generating feature importance plot...")
    plot_feature_importance(xgb_model, feature_cols)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best Model: XGBoost with AUROC = {results[0]['AUROC']:.4f}")
    print(f"Improvement over qSOFA: +{(results[0]['AUROC'] - qsofa_auroc)*100:.1f}%")
    
    return xgb_model, feature_cols, results


def generate_synthetic_data(n_patients: int = 500) -> pd.DataFrame:
    """
    Generate synthetic data for testing when real data is unavailable.
    """
    np.random.seed(42)
    records = []
    
    for pid in range(n_patients):
        n_hours = np.random.randint(20, 100)
        has_sepsis = np.random.random() < 0.15  # 15% sepsis rate
        sepsis_hour = np.random.randint(n_hours//2, n_hours) if has_sepsis else n_hours + 10
        
        for hour in range(n_hours):
            # Generate vital signs with trends for sepsis patients
            decay = 1 + 0.02 * max(0, hour - sepsis_hour + 6) if has_sepsis else 0
            
            record = {
                'patient_id': f'p{pid:05d}',
                'hour': hour,
                'HR': np.clip(80 + np.random.randn() * 10 + decay * 15, 40, 180),
                'O2Sat': np.clip(98 - np.random.randn() * 2 - decay * 3, 70, 100),
                'Temp': np.clip(37.0 + np.random.randn() * 0.5 + decay * 0.8, 35, 42),
                'SBP': np.clip(120 - np.random.randn() * 10 - decay * 20, 60, 200),
                'MAP': np.clip(90 - np.random.randn() * 8 - decay * 15, 40, 150),
                'DBP': np.clip(80 - np.random.randn() * 6 - decay * 10, 40, 120),
                'Resp': np.clip(16 + np.random.randn() * 3 + decay * 5, 8, 40),
                'EtCO2': np.clip(35 + np.random.randn() * 3, 20, 60),
                'WBC': np.clip(8 + np.random.randn() * 2 + decay * 5, 2, 30),
                'Lactate': np.clip(1.5 + np.random.randn() * 0.5 + decay * 2, 0.5, 10),
                'Creatinine': np.clip(1.0 + np.random.randn() * 0.3 + decay * 0.5, 0.3, 8),
                'Platelets': np.clip(250 - np.random.randn() * 50 - decay * 50, 20, 500),
                'Bilirubin_total': np.clip(0.8 + np.random.randn() * 0.2 + decay * 0.5, 0.1, 15),
                'Age': 50 + np.random.randn() * 15,
                'Gender': np.random.randint(0, 2),
                'Unit1': np.random.randint(0, 2),
                'Unit2': np.random.randint(0, 2),
                'HospAdmTime': -np.random.randint(0, 100),
                'ICULOS': hour + 1,
                'SepsisLabel': 1 if (has_sepsis and hour >= sepsis_hour - 6) else 0,
            }
            
            # Add remaining lab values with high missing rate
            for lab in LAB_VALUES:
                if lab not in record:
                    record[lab] = np.nan if np.random.random() < 0.85 else np.random.randn() * 10 + 50
            
            records.append(record)
    
    return pd.DataFrame(records)


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    model, features, results = main()
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR YOUR HACKATHON:")
    print("="*60)
    print("""
    1. Download real PhysioNet 2019 data
    2. Tune hyperparameters (try different max_depth, n_estimators)
    3. Add SHAP visualizations: 
       - pip install shap
       - See explain_predictions() function
    4. Build Streamlit dashboard:
       - pip install streamlit
       - streamlit run app.py
    5. Compare with SOFA score (calculate from raw features)
    6. Add patient timeline visualization
    
    Good luck! 🏆
    """)
