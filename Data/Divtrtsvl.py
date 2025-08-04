# split_dataset.py

import os
import random
import shutil

def split_fragments_folder(base_folder: str, splits: dict, seed: int = 42) -> None:
    """
    Divide los archivos CSV en subcarpetas 'train', 'val' y 'test' bajo base_folder,
    según las proporciones definidas en `splits`.
    """
    # 1. Crear o reiniciar carpetas de split
    for split_name in splits:
        split_dir = os.path.join(base_folder, split_name)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)

    # 2. Listar todos los archivos CSV (solo en la carpeta base, no en subcarpetas)
    all_files = [
        os.path.join(base_folder, f)
        for f in os.listdir(base_folder)
        if f.lower().endswith('.csv') and os.path.isfile(os.path.join(base_folder, f))
    ]

    # 3. Barajar con semilla fija para reproducibilidad
    random.seed(seed)
    random.shuffle(all_files)

    # 4. Calcular counts para cada split
    total = len(all_files)
    n_train = int(total * splits['train'])
    n_val   = int(total * splits['val'])
    n_test  = total - n_train - n_val  # el resto va a test

    # 5. Asignar archivos a cada split
    assignments = {
        'train': all_files[:n_train],
        'val':   all_files[n_train:n_train + n_val],
        'test':  all_files[n_train + n_val:]
    }

    # 6. Mover archivos a sus carpetas correspondientes
    for split_name, files in assignments.items():
        dest_dir = os.path.join(base_folder, split_name)
        for src in files:
            shutil.move(src, dest_dir)

    # 7. Informe
    print("✅ Distribución aleatoria de archivos completa.")
    print(f"• Train: {n_train} archivos → {os.path.join(base_folder, 'train')}")
    print(f"• Val:   {n_val} archivos → {os.path.join(base_folder, 'val')}")
    print(f"• Test:  {n_test} archivos → {os.path.join(base_folder, 'test')}")

def main():
    # Carpeta donde están los CSV de fragmentos separados
    base_folder = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dta\df_separados"

    # Proporciones para train/val/test
    splits = {
        'train': 0.50,
        'val':   0.40,
        'test':  0.10
    }

    split_fragments_folder(base_folder, splits)

if __name__ == "__main__":
    main()
