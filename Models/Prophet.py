# train_prophet.py

import os
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

def load_train_data(train_dir: str) -> pd.DataFrame:
    """
    Carga y concatena todos los CSV de entrenamiento en train_dir,
    parseando la columna 'ds' como datetime.
    """
    csv_files = [
        os.path.join(train_dir, f)
        for f in os.listdir(train_dir)
        if f.lower().endswith('.csv')
    ]
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {train_dir}")
    df_list = [pd.read_csv(fp, parse_dates=['ds']) for fp in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Cargados {len(csv_files)} archivos de entrenamiento, total filas: {df.shape[0]}")
    return df

def train_and_evaluate(train_df: pd.DataFrame,
                       initial: str = '365 days',
                       period:  str = '90 days',
                       horizon: str = '90 days') -> None:
    """
    Entrena un modelo Prophet sobre train_df y realiza validación cruzada,
    imprime métricas y grafica MAE, RMSE y SMAPE vs horizonte.
    """
    # 1. Inicializar y entrenar
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5
    )
    m.fit(train_df)
    print("✅ Modelo Prophet entrenado.")

    # 2. Validación cruzada
    df_cv = cross_validation(
        m,
        initial=initial,
        period=period,
        horizon=horizon,
        parallel="processes"
    )
    print(f"✅ Validación cruzada completada: {df_cv.shape[0]} filas")

    # 3. Métricas de desempeño
    df_p = performance_metrics(df_cv)
    print("✅ Cálculo de métricas de desempeño:")
    print(df_p[['horizon','mae','rmse','smape']].to_string(index=False))

    # 4. Graficar MAE y RMSE vs horizonte
    x = df_p['horizon'].dt.days
    plt.figure()
    plt.plot(x, df_p['mae'],  marker='o', label='MAE')
    plt.plot(x, df_p['rmse'], marker='o', label='RMSE')
    plt.xlabel('Horizonte (días)')
    plt.ylabel('Error')
    plt.title('MAE y RMSE vs Horizonte de pronóstico')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. Graficar SMAPE vs horizonte
    plt.figure()
    plt.plot(x, df_p['smape'], marker='o')
    plt.xlabel('Horizonte (días)')
    plt.ylabel('SMAPE')
    plt.title('SMAPE vs Horizonte de pronóstico')
    plt.tight_layout()
    plt.show()

def main():
    # Ruta a la carpeta 'train' dentro de df_separados
    base = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dta\df_separados"
    train_dir = os.path.join(base, 'train')

    # 1) Cargar datos de entrenamiento
    train_df = load_train_data(train_dir)

    # 2) Entrenar y evaluar Prophet
    train_and_evaluate(
        train_df,
        initial='365 days',
        period='90 days',
        horizon='90 days'
    )

if __name__ == "__main__":
    main()
