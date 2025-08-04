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

def main():
    # 1. Ajusta esta ruta a donde tengas tu CSV en Windows
    ruta_csv = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dataset_combinado.csv"

    # 2. Carga y muestra las primeras filas
    df = load_dataset(ruta_csv)
    print(df.head())

if __name__ == "__main__":
    main()
