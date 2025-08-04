# train_lstm_gpu.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 0. Configurar TensorFlow para usar GPU (si está disponible)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Permitir crecimiento de memoria
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow GPU available: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠️ Error configuring GPU: {e}")
else:
    print("⚠️ No GPU found, running on CPU.")

# 1. Rutas de datos
BASE_DIR   = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dta\df_separados"
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VAL_DIR    = os.path.join(BASE_DIR, "val")
TEST_DIR   = os.path.join(BASE_DIR, "test")

# 2. Cargar listas de archivos
train_files = glob.glob(os.path.join(TRAIN_DIR, "*.csv"))
val_files   = glob.glob(os.path.join(VAL_DIR,   "*.csv"))
test_files  = glob.glob(os.path.join(TEST_DIR,  "*.csv"))

def load_split(file_list):
    """Carga y concatena ds,y de una lista de CSVs."""
    dfs = []
    for f in file_list:
        df = pd.read_csv(f, parse_dates=['ds'])
        dfs.append(df[['ds', 'y']])
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values('ds').reset_index(drop=True)

# 3. Cargar splits
train_df = load_split(train_files)
val_df   = load_split(val_files)
test_df  = load_split(test_files)

# 4. Escalado con MinMaxScaler (fit solo en train)
scaler = MinMaxScaler()
train_df['y_s'] = scaler.fit_transform(train_df[['y']])
val_df['y_s']   = scaler.transform(val_df[['y']])
test_df['y_s']  = scaler.transform(test_df[['y']])

# 5. Generar secuencias
SEQ_LEN = 50

def make_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    return X, np.array(y)

X_train, y_train = make_sequences(train_df['y_s'].values, SEQ_LEN)
X_val,   y_val   = make_sequences(val_df['y_s'].values,   SEQ_LEN)
X_test,  y_test  = make_sequences(test_df['y_s'].values,  SEQ_LEN)

print("Shapes →",
      "X_train:", X_train.shape,
      "X_val:",   X_val.shape,
      "X_test:",  X_test.shape)

# 6. Definir modelo LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.1),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 7. Callbacks: EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 8. Entrenamiento
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# 9. Graficar métricas de entrenamiento
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'],   label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('MSE durante entrenamiento')
plt.xlabel('Época'); plt.ylabel('MSE'); plt.legend(); plt.grid(); plt.show()

plt.figure(figsize=(12, 4))
plt.plot(history.history['mae'],   label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('MAE durante entrenamiento')
plt.xlabel('Época'); plt.ylabel('MAE'); plt.legend(); plt.grid(); plt.show()

# 10. Pronóstico en test
y_pred_s = model.predict(X_test)
y_pred   = scaler.inverse_transform(y_pred_s)
y_true   = scaler.inverse_transform(y_test.reshape(-1, 1))
dates_test = test_df['ds'].iloc[SEQ_LEN:].values

plt.figure(figsize=(12, 4))
plt.plot(dates_test, y_true.flatten(), 'g-',  label='Real Test')
plt.plot(dates_test, y_pred.flatten(), 'r--', label='Forecast LSTM')
plt.title('Pronóstico vs Real en Test')
plt.xlabel('Fecha'); plt.ylabel('Ventas'); plt.legend(); plt.grid(); plt.show()

# 11. Métricas finales
mae_val = mean_absolute_error(
    scaler.inverse_transform(y_val.reshape(-1,1)),
    scaler.inverse_transform(model.predict(X_val))
)
mae_test = mean_absolute_error(y_true, y_pred)

print(f"MAE en validación (real): {mae_val:.2f}")
print(f"MAE en test (real):       {mae_test:.2f}")
