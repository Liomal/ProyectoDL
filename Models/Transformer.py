# train_transformer.py

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# 0. Forzar uso de GPU si está disponible (CUDA), sino CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Configuración de rutas y parámetros
BASE_DIR   = r"C:\Users\USUARIO\OneDrive\Escritorio\Proy final DL\dta\df_separados"
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VAL_DIR    = os.path.join(BASE_DIR, "val")
TEST_DIR   = os.path.join(BASE_DIR, "test")

LOOKBACK   = 28
HORIZON    = 7
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-3

# 2. Normalización global con StandardScaler (ajustado sólo sobre train)
def fit_scaler(train_dir):
    files = glob.glob(os.path.join(train_dir, "*.csv"))
    dfs = [pd.read_csv(f, parse_dates=['ds'])[['y']] for f in files]
    all_train = pd.concat(dfs, ignore_index=True)
    scaler = StandardScaler().fit(all_train[['y']])
    return scaler

# 3. Dataset para series temporales
class TimeSeriesDataset(Dataset):
    def __init__(self, files, scaler):
        self.samples = []
        for f in files:
            df = pd.read_csv(f, parse_dates=['ds'])
            y = scaler.transform(df[['y']]).flatten()
            for i in range(len(y) - LOOKBACK - HORIZON + 1):
                seq = y[i : i + LOOKBACK]
                tgt = y[i + LOOKBACK : i + LOOKBACK + HORIZON]
                self.samples.append((seq, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, tgt = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

# 4. Modelo Transformer para forecasting
class TimeSeriesTransformer(nn.Module):
    def __init__(self, lookback, horizon, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=128)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.output_head = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (batch, lookback)
        x = x.unsqueeze(-1)                  # (batch, lookback, 1)
        x = self.input_proj(x)               # (batch, lookback, d_model)
        x = x.permute(1, 0, 2)               # (lookback, batch, d_model)
        encoded = self.encoder(x)            # (lookback, batch, d_model)
        out = encoded[-1, :, :]              # (batch, d_model)
        return self.output_head(out)         # (batch, horizon)

def make_dataloader(dir_path, scaler, shuffle=False):
    files = glob.glob(os.path.join(dir_path, "*.csv"))
    ds = TimeSeriesDataset(files, scaler)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

def train():
    # Ajustar scaler
    scaler = fit_scaler(TRAIN_DIR)

    # Preparar DataLoaders
    train_loader = make_dataloader(TRAIN_DIR, scaler, shuffle=True)
    val_loader   = make_dataloader(VAL_DIR,   scaler, shuffle=False)
    test_loader  = make_dataloader(TEST_DIR,  scaler, shuffle=False)

    # Inicializar modelo, optimizador y función de pérdida
    model = TimeSeriesTransformer(LOOKBACK, HORIZON).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                p = model(xb).cpu().numpy().flatten()
                t = yb.numpy().flatten()
                val_preds.append(p)
                val_trues.append(t)
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        # Invertir escala
        val_preds_inv = scaler.inverse_transform(val_preds.reshape(-1,1)).flatten()
        val_trues_inv = scaler.inverse_transform(val_trues.reshape(-1,1)).flatten()
        val_mae = mean_absolute_error(val_trues_inv, val_preds_inv)

        print(f"Epoch {epoch}: Train Loss {np.mean(train_losses):.4f}, Val MAE {val_mae:.2f}")

    # Test final
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p = model(xb).cpu().numpy().flatten()
            t = yb.numpy().flatten()
            test_preds.append(p)
            test_trues.append(t)
    test_preds = np.concatenate(test_preds)
    test_trues = np.concatenate(test_trues)
    test_preds_inv = scaler.inverse_transform(test_preds.reshape(-1,1)).flatten()
    test_trues_inv = scaler.inverse_transform(test_trues.reshape(-1,1)).flatten()
    test_mae = mean_absolute_error(test_trues_inv, test_preds_inv)
    print(f"✅ Test MAE: {test_mae:.2f}")

if __name__ == "__main__":
    train()
