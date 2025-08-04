# main.py

import os
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga un CSV desde la ruta dada y comprueba que exista.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset no encontrado en: {path}")
    df = pd.read_csv(path)
    print(f"Dimensiones del DataFrame: {df.shape}")
    return df

def process_data(df: pd.DataFrame, base_dir: str) -> None:
    """
    Convierte la columna Fecha a datetime, agrupa, filtra y guarda
    ventas_diarias_familia_sucursal_filtrado.csv en base_dir/filtrado.
    """
    # 1. Convertir Fecha a datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # 2. Agrupar por Familia, Sucursal, Fecha
    ventas_diarias = (
        df
        .groupby(['Familia', 'Sucursal', 'Fecha'])['Venta neta s/IVA']
        .sum()
        .reset_index()
    )

    # 3. Contar días únicos
    conteo_dias = (
        ventas_diarias
        .groupby(['Familia', 'Sucursal'])['Fecha']
        .nunique()
        .reset_index(name='data_points')
    )

    # 4. Filtrar >100 días
    validos = conteo_dias[conteo_dias['data_points'] > 100]

    # 5. Merge y rename
    ventas_filtradas = ventas_diarias.merge(
        validos[['Familia', 'Sucursal']],
        on=['Familia', 'Sucursal'],
        how='inner'
    ).rename(columns={'Fecha': 'ds', 'Venta neta s/IVA': 'y'})

    # 6. Guardar en subcarpeta 'filtrado'
    out_dir = os.path.join(base_dir, 'filtrado')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ventas_diarias_familia_sucursal_filtrado.csv')
    ventas_filtradas.to_csv(out_path, index=False)

    print("✅ Datos diarios preparados.")
    print(f"  Guardado en: {out_path}")
    print(f"  Combinaciones ≥100 días: {validos.shape[0]}")
    print(f"  Filas totales: {ventas_filtradas.shape[0]}")

def fragment_data(input_csv: str, base_dir: str) -> None:
    """
    Toma el CSV filtrado y genera:
      - base_dir/fragmentos/fragmentos_continuos_ventas_familia_sucursal.csv
    """
    df = pd.read_csv(input_csv, parse_dates=['ds'])
    fragmentos = []

    for (familia, sucursal), grupo in df.groupby(['Familia', 'Sucursal']):
        grupo = grupo.sort_values('ds').reset_index(drop=True)
        grupo['gap'] = grupo['ds'].diff().dt.days.fillna(1).astype(int)
        grupo['fragment_id'] = (grupo['gap'] > 1).cumsum()

        for _, frag in grupo.groupby('fragment_id'):
            if len(frag) >= 100:
                frag = frag.drop(columns=['gap', 'fragment_id'])
                frag['Familia'], frag['Sucursal'] = familia, sucursal
                fragmentos.append(frag)

    df_frag = pd.concat(fragmentos, ignore_index=True)

    out_dir = os.path.join(base_dir, 'fragmentos')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fragmentos_continuos_ventas_familia_sucursal.csv')
    df_frag.to_csv(out_path, index=False)

    print("✅ Fragmentos continuos generados.")
    print(f"  Guardado en: {out_path}")
    print(f"  Fragmentos válidos: {len(fragmentos)}")
    print(f"  Filas totales: {df_frag.shape[0]}")

def split_fragments(input_csv: str, base_dir: str) -> None:
    """
    Lee el CSV de todos los fragmentos continuos y crea un CSV
    independiente por fragmento en base_dir/df_separados.
    """
    df = pd.read_csv(input_csv, parse_dates=['ds'])
    out_dir = os.path.join(base_dir, 'df_separados')
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for (familia, sucursal), grupo in df.groupby(['Familia', 'Sucursal']):
        grupo = grupo.sort_values('ds').reset_index(drop=True)
        grupo['gap'] = grupo['ds'].diff().dt.days.fillna(1).astype(int)
        grupo['fragment_id'] = (grupo['gap'] > 1).cumsum()

        for frag_id, frag in grupo.groupby('fragment_id'):
            if len(frag) >= 100:
                frag = frag.drop(columns=['gap', 'fragment_id'])
                frag['Familia'], frag['Sucursal'] = familia, sucursal
                filename = (
                    f"{familia.replace(' ', '_')}_"
                    f"{sucursal.replace(' ', '_')}_frag{frag_id}.csv"
                )
                frag.to_csv(os.path.join(out_dir, filename), index=False)
                count += 1

    print("✅ CSVs individuales creados.")
    print(f"  Total archivos: {count}")
    print(f"  Directorio:   {out_dir}")

def main():
    # carpeta base 'dta'
    base = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dta"
    # CSV original
    original_csv = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dataset_combinado.csv"

    # 1) Agregación y filtrado
    df = load_dataset(original_csv)
    process_data(df, base)

    # 2) Fragmentación continua
    filtered_csv = os.path.join(base, 'filtrado', 'ventas_diarias_familia_sucursal_filtrado.csv')
    fragment_data(filtered_csv, base)

    # 3) Split en CSVs
    frag_csv = os.path.join(base, 'fragmentos', 'fragmentos_continuos_ventas_familia_sucursal.csv')
    split_fragments(frag_csv, base)

if __name__ == "__main__":
    main()
