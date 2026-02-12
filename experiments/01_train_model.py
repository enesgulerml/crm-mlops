import pandas as pd
import numpy as np
import yaml
import os
import xgboost as xgb
import optuna

# Metrics & Sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

# ONNX
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost


# --- CONFIG LOADING ---
def load_config():
    config_path = "config/params.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


params = load_config()

# --- XGBOOST REGISTRATION ---
try:
    update_registered_converter(
        xgb.XGBClassifier,
        'XGBoostXGBClassifier',
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
    )
except Exception:
    pass


# --- DATA ---
def load_and_clean_data(filepath):
    print(f"Loading data: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    if 'customerID' in df.columns: df = df.drop(['customerID'], axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    for col in df.select_dtypes(include=['object']).columns:
        if col != params['data']['target_col']:
            df[col] = df[col].fillna("Missing").astype(str)
    return df


def feature_engineering(df):
    df_fe = df.copy()
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    existing = [col for col in services if col in df_fe.columns]
    for col in existing:
        df_fe[col + '_Flag'] = df_fe[col].apply(lambda x: 1 if 'Yes' in str(x) else 0)

    if existing:
        flag_cols = [c + '_Flag' for c in existing]
        df_fe['TotalServices'] = df_fe[flag_cols].sum(axis=1)
        df_fe['AvgChargePerService'] = df_fe['MonthlyCharges'] / (df_fe['TotalServices'] + 1)
        df_fe = df_fe.drop(flag_cols, axis=1)
    return df_fe


def get_pipeline(X_train):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, X_train.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', categorical_transformer, X_train.select_dtypes(include=['object']).columns)
    ])
    return preprocessor, X_train.select_dtypes(include=['object']).columns


# --- OPTIMIZATION ---
def objective(trial, X, y, preprocessor):
    param = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    }
    model = xgb.XGBClassifier(**param, random_state=params['base']['random_state'])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    try:
        return cross_val_score(pipeline, X, y, cv=params['training']['cv'], scoring='roc_auc').mean()
    except:
        return 0.0


# --- MAIN ---
def main():
    # 1. Prep
    print(f"Configuration loaded: config/params.yaml")
    df = load_and_clean_data(params['data']['source'])
    df = feature_engineering(df)

    target_col = params['data']['target_col']
    print(f"Encoding the target ({target_col})...")
    df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['base']['test_size'], stratify=y,
                                                        random_state=42)

    # 2. Train
    preprocessor, cat_cols = get_pipeline(X_train)
    print("Optuna is working...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, X_train, y_train, preprocessor), n_trials=params['training']['n_trials'])

    print("Final Model...")
    final_model = xgb.XGBClassifier(**study.best_trial.params, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_model)])
    pipeline.fit(X_train, y_train)

    # 3. Evaluation
    print("\n" + "=" * 40)
    print("MODEL PERFORMANCE REPORT")
    print("=" * 40)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("=" * 40 + "\n")

    # 4. Export
    print("ONNX Export...")
    os.makedirs(os.path.dirname(params['model']['output_path']), exist_ok=True)
    initial_types = [(c, StringTensorType([None, 1])) if c in cat_cols else (c, FloatTensorType([None, 1])) for c in
                     X.columns]

    try:
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types, target_opset={'': 12, 'ai.onnx.ml': 3})
        with open(params['model']['output_path'], "wb") as f:
            f.write(onnx_model.SerializeToString())

        size_kb = os.path.getsize(params['model']['output_path']) / 1024
        print(f"SUCCESSFUL! Registration: {params['model']['output_path']}")
        print(f"Model Size: {size_kb:.2f} KB")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()