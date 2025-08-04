# main.py

import os
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset no encontrado en: {path}")
    df = pd.read_csv(path)
    print(f"Dimensiones del DataFrame: {df.shape}")
    return df

def process_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Convierte fechas, agrupa, filtra y guarda el CSV resultante.
    """
    # 5. Convertir la columna Fecha a datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # 6. Agrupar por Familia, Sucursal y Fecha (ventas diarias)
    ventas_diarias = (
        df
        .groupby(['Familia', 'Sucursal', 'Fecha'])['Venta neta s/IVA']
        .sum()
        .reset_index()
    )

    # 7. Contar días únicos por Familia y Sucursal
    conteo_dias = (
        ventas_diarias
        .groupby(['Familia', 'Sucursal'])['Fecha']
        .nunique()
        .reset_index(name='data_points')
    )

    # 8. Filtrar las combinaciones con más de 100 días de datos
    validos = conteo_dias[conteo_dias['data_points'] > 100]

    # 9. Quedarse solo con esas combinaciones
    ventas_filtradas = ventas_diarias.merge(
        validos[['Familia', 'Sucursal']],
        on=['Familia', 'Sucursal'],
        how='inner'
    )

    # 10. Renombrar columnas para Prophet
    ventas_filtradas.rename(
        columns={'Fecha': 'ds', 'Venta neta s/IVA': 'y'},
        inplace=True
    )

    # Asegurarse de que la carpeta exista
    os.makedirs(output_dir, exist_ok=True)

    # 11. Guardar el resultado en local
    output_path = os.path.join(
        output_dir,
        'ventas_diarias_familia_sucursal_filtrado.csv'
    )
    ventas_filtradas.to_csv(output_path, index=False)

    # 12. Mensajes de salida
    print("✅ Datos diarios por Familia-Sucursal preparados.")
    print(f"📁 Guardado en: {output_path}")
    print(f"🔢 Combinaciones Familia-Sucursal con >100 días: {validos.shape[0]}")
    print(f"📊 Total de filas en dataset final: {ventas_filtradas.shape[0]}")

def main():
    # Ruta al CSV original
    ruta_csv = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dataset_combinado.csv"
    # Carpeta de salida
    carpeta_salida = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dta"

    # 1. Cargar
    df = load_dataset(ruta_csv)
    # 2. Procesar y guardar
    process_data(df, carpeta_salida)

if __name__ == "__main__":
    main()
