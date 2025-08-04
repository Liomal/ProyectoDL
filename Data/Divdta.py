import os
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset no encontrado en: {path}")
    df = pd.read_csv(path)
    print(f"Dimensiones del DataFrame: {df.shape}")
    return df

def process_data(df: pd.DataFrame, output_dir: str) -> None:
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

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,
        'ventas_diarias_familia_sucursal_filtrado.csv'
    )
    ventas_filtradas.to_csv(output_path, index=False)

    print("✅ Datos diarios por Familia-Sucursal preparados.")
    print(f"📁 Guardado en: {output_path}")
    print(f"🔢 Combinaciones Familia-Sucursal con >100 días: {validos.shape[0]}")
    print(f"📊 Total de filas en dataset final: {ventas_filtradas.shape[0]}")

def fragment_data(input_csv: str, output_dir: str) -> None:
    # 1. Carga, parseando ds como fecha
    df = pd.read_csv(input_csv, parse_dates=['ds'])

    fragmentos_validos = []

    # 2. Recorre cada Familia+Sucursal
    for (familia, sucursal), grupo in df.groupby(['Familia', 'Sucursal']):
        grupo = grupo.sort_values('ds').reset_index(drop=True)

        # 2.1 Gap en días
        grupo['gap'] = grupo['ds'].diff().dt.days.fillna(1).astype(int)
        # 2.2 Fragment ID (nuevo cuando gap>1)
        grupo['fragment_id'] = (grupo['gap'] > 1).cumsum()

        # 2.3 Extrae fragmentos >=100 filas
        for _, frag in grupo.groupby('fragment_id'):
            if len(frag) >= 100:
                frag = frag.drop(columns=['gap', 'fragment_id'])
                frag['Familia']  = familia
                frag['Sucursal'] = sucursal
                fragmentos_validos.append(frag)

    # 3. Concatenar y guardar
    df_fragmentos = pd.concat(fragmentos_validos, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,
        'fragmentos_continuos_ventas_familia_sucursal.csv'
    )
    df_fragmentos.to_csv(output_path, index=False)

    print("✅ Fragmentación de series realizada.")
    print(f"📁 Fragmentos guardados en: {output_path}")
    print(f"🔢 Total de fragmentos válidos: {len(fragmentos_validos)}")
    print(f"📊 Total de filas en el dataset final: {df_fragmentos.shape[0]}")

def main():
    # Rutas
    ruta_csv            = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dataset_combinado.csv"
    carpeta_salida       = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dta"

    # 1) Carga y procesamiento inicial
    df = load_dataset(ruta_csv)
    process_data(df, carpeta_salida)

    # 2) Fragmentación sobre el CSV que acabamos de generar
    input_filtrado = os.path.join(
        carpeta_salida,
        'ventas_diarias_familia_sucursal_filtrado.csv'
    )
    fragment_data(input_filtrado, carpeta_salida)

if __name__ == "__main__":
    main()
