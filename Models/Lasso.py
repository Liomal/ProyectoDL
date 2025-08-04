# train_lasso.py

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

def create_features(df, lags=[1, 7, 14], rolls=[7, 14]):
    """
    Genera variables rezagadas, medias móviles y dummies de día de semana.
    """
    X = pd.DataFrame()
    for lag in lags:
        X[f'lag_{lag}'] = df['y'].shift(lag)
    for roll in rolls:
        X[f'roll_mean_{roll}'] = df['y'].shift(1).rolling(roll).mean()
    dow = df['ds'].dt.dayofweek
    dummies = pd.get_dummies(dow, prefix='dow', drop_first=True)
    X = pd.concat([X, dummies], axis=1)
    return X.fillna(0)

def load_split_data(split_dir: str) -> pd.DataFrame:
    """
    Carga y concatena todos los CSV de un split (train o val).
    """
    files = glob.glob(os.path.join(split_dir, '*.csv'))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {split_dir}")
    dfs = [pd.read_csv(f, parse_dates=['ds']) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('ds').reset_index(drop=True)
    print(f"Cargados {len(files)} archivos de {split_dir}, filas totales: {df.shape[0]}")
    return df

def train_lasso(train_df: pd.DataFrame) -> Pipeline:
    """
    Ajusta un Pipeline con StandardScaler + LassoCV (CV temporal).
    """
    X_train = create_features(train_df)
    y_train = train_df['y']
    tscv = TimeSeriesSplit(n_splits=5)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(cv=tscv, random_state=42, max_iter=5000))
    ])
    pipeline.fit(X_train, y_train)
    print("✅ LassoCV entrenado.")
    coefs = pd.Series(pipeline.named_steps['lasso'].coef_, index=X_train.columns)
    print("Coeficientes no nulos:")
    print(coefs[coefs != 0])
    return pipeline

def evaluate_lasso(pipeline: Pipeline, val_df: pd.DataFrame) -> None:
    """
    Calcula MAE sobre el conjunto de validación y reporta promedio ± std.
    """
    X_val = create_features(val_df)
    y_val = val_df['y']
    preds = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"MAE validación (Lasso): {mae:.2f}")

def plot_validation_fragments(pipeline: Pipeline, val_dir: str) -> None:
    """
    Para cada fragmento en val_dir, grafica la serie real vs la predicción.
    """
    val_files = glob.glob(os.path.join(val_dir, '*.csv'))
    if not val_files:
        print(f"No hay archivos CSV en {val_dir} para graficar.")
        return

    for file_path in val_files:
        df_true = pd.read_csv(file_path, parse_dates=['ds'])
        Xv = create_features(df_true)
        y_pred = pipeline.predict(Xv)

        plt.figure(figsize=(10, 4))
        plt.plot(df_true['ds'], df_true['y'], label='Real')
        plt.plot(df_true['ds'], y_pred,  label='Predicha')
        plt.title(f'Predicción vs Real: {os.path.basename(file_path)}')
        plt.xlabel('Fecha')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    base = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dta\df_separados"
    train_dir = os.path.join(base, 'train')
    val_dir   = os.path.join(base, 'val')

    # 1) Cargar splits
    train_df = load_split_data(train_dir)
    val_df   = load_split_data(val_dir)

    # 2) Entrenar Lasso
    model = train_lasso(train_df)

    # 3) Evaluar en validación
    evaluate_lasso(model, val_df)

    # 4) Graficar fragmentos de validación
    plot_validation_fragments(model, val_dir)

if __name__ == "__main__":
    main()
