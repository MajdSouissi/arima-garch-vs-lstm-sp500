"""
=============================================================================
06_lstm_models.py
LSTM Building, Training, and Out-of-Sample Forecasting (PyTorch)
=============================================================================
Builds and trains three LSTM models:
  1. Base LSTM (Target: Return, Features: Log_Return only)
  2. Enhanced LSTM (Target: Return, Features: Return + r^2 + Vol + MAE)
  3. Volatility LSTM (Target: r^2, Features: Return + r^2 + Vol + MAE)

Uses early stopping on the validation set. Saves forecasts for test set.
Framework: PyTorch (avoids TensorFlow/AVX dependency issues).
=============================================================================
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys
import random
import time

# â”€â”€ PyTorch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# â”€â”€ 1. Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

BASE_DIR = str(Path(__file__).resolve().parent.parent)
DATA_DIR   = os.path.join(BASE_DIR, "lstm_data")
OUTPUT_CSV = os.path.join(BASE_DIR, "lstm_forecasts.csv")
MODELS_DIR = os.path.join(BASE_DIR, "lstm_saved_models")
PLOTS_DIR  = os.path.join(BASE_DIR, "lstm_plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# â”€â”€ 2. Load Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nLoading LSTM sequences...")

X_train_base = np.load(os.path.join(DATA_DIR, 'seq_X_train_base.npy'))
y_train_ret  = np.load(os.path.join(DATA_DIR, 'seq_y_train_ret.npy'))
X_val_base   = np.load(os.path.join(DATA_DIR, 'seq_X_val_base.npy'))
y_val_ret    = np.load(os.path.join(DATA_DIR, 'seq_y_val_ret.npy'))
X_test_base  = np.load(os.path.join(DATA_DIR, 'seq_X_test_base.npy'))
y_test_ret   = np.load(os.path.join(DATA_DIR, 'seq_y_test_ret.npy'))

X_train_enh  = np.load(os.path.join(DATA_DIR, 'seq_X_train_enh.npy'))
y_train_vol  = np.load(os.path.join(DATA_DIR, 'seq_y_train_vol.npy'))
X_val_enh    = np.load(os.path.join(DATA_DIR, 'seq_X_val_enh.npy'))
y_val_vol    = np.load(os.path.join(DATA_DIR, 'seq_y_val_vol.npy'))
X_test_enh   = np.load(os.path.join(DATA_DIR, 'seq_X_test_enh.npy'))
y_test_vol   = np.load(os.path.join(DATA_DIR, 'seq_y_test_vol.npy'))

test_dates   = np.load(os.path.join(DATA_DIR, 'test_dates.npy'), allow_pickle=True)

print(f"X_train_base: {X_train_base.shape}  y_train_ret: {y_train_ret.shape}")
print(f"X_train_enh:  {X_train_enh.shape}  y_train_vol: {y_train_vol.shape}")
print(f"X_val_base:   {X_val_base.shape}   y_val_ret:   {y_val_ret.shape}")
print(f"X_test_base:  {X_test_base.shape}  y_test_ret:  {y_test_ret.shape}")
print(f"X_test_enh:   {X_test_enh.shape}   y_test_vol:  {y_test_vol.shape}")
print(f"Test dates:   {len(test_dates)}")

# â”€â”€ 3. PyTorch LSTM Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class LSTMForecaster(nn.Module):
    """
    Single-layer LSTM â†’ Dropout â†’ Linear(1).
    Parsimonious architecture appropriate for daily return forecasting.
    """
    def __init__(self, input_size, hidden_size=32, dropout=0.2, is_volatility=False):
        super().__init__()
        self.is_volatility = is_volatility
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0  # dropout between layers; we use 1 layer so apply manually
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]    # (batch, hidden) â€” last timestep
        out = self.dropout(last_hidden)
        out = self.fc(out)                  # (batch, 1)
        if self.is_volatility:
            out = self.softplus(out) + 1e-6
        return out.squeeze(-1)              # (batch,)


# â”€â”€ 4. Training Function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=100, batch_size=64, lr=0.001, patience=10,
                model_name="model"):
    """
    Train with Adam + MSE loss + early stopping on validation loss.
    Returns: (model, history_dict)
    """
    model.to(DEVICE)

    # Create DataLoaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\n{'Epoch':>6s}  {'Train Loss':>12s}  {'Val Loss':>12s}  {'Status':>12s}")
    print(f"{'â”€'*6}  {'â”€'*12}  {'â”€'*12}  {'â”€'*12}")

    for epoch in range(1, epochs + 1):
        # â”€â”€ Training â”€â”€
        model.train()
        epoch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)

        # â”€â”€ Validation â”€â”€
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # â”€â”€ Early stopping â”€â”€
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
            status = "âœ“ improved"
        else:
            patience_counter += 1
            status = f"  patience {patience_counter}/{patience}"

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0:
            print(f"{epoch:6d}  {train_loss:12.6f}  {val_loss:12.6f}  {status}")

        if patience_counter >= patience:
            print(f"{epoch:6d}  {train_loss:12.6f}  {val_loss:12.6f}  EARLY STOP")
            break

    # Restore best weights
    model.load_state_dict(best_state)

    # Save model
    save_path = os.path.join(MODELS_DIR, f'{model_name}.pt')
    torch.save(best_state, save_path)
    print(f"Best val_loss: {best_val_loss:.6f} (epoch {np.argmin(history['val_loss'])+1})")
    print(f"Model saved: {save_path}")

    return model, history


def predict(model, X):
    """Generate predictions from a trained model."""
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE)
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    return pred


# â”€â”€ 5. Hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
HIDDEN   = 32
DROPOUT  = 0.2
LR       = 0.001
BATCH    = 64
EPOCHS   = 100
PATIENCE = 10

# â”€â”€ 6. Train Model 1 â€” Return | Base Features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "="*70)
print("  MODEL 1: Return Target | Base Features (Log_Return only)")
print("="*70)

set_seeds()
model1 = LSTMForecaster(input_size=X_train_base.shape[2],
                         hidden_size=HIDDEN, dropout=DROPOUT, is_volatility=False)
print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")

model1, hist1 = train_model(
    model1, X_train_base, y_train_ret, X_val_base, y_val_ret,
    epochs=EPOCHS, batch_size=BATCH, lr=LR, patience=PATIENCE,
    model_name='model1_base_return'
)

# â”€â”€ 7. Train Model 2 â€” Return | Enhanced Features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "="*70)
print("  MODEL 2: Return Target | Enhanced Features")
print("="*70)

set_seeds()
model2 = LSTMForecaster(input_size=X_train_enh.shape[2],
                         hidden_size=HIDDEN, dropout=DROPOUT, is_volatility=False)
print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")

model2, hist2 = train_model(
    model2, X_train_enh, y_train_ret, X_val_enh, y_val_ret,
    epochs=EPOCHS, batch_size=BATCH, lr=LR, patience=PATIENCE,
    model_name='model2_enh_return'
)

# â”€â”€ 8. Train Model 3 â€” Volatility | Enhanced Features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "="*70)
print("  MODEL 3: Volatility Target (r^2) | Enhanced Features")
print("="*70)

set_seeds()
model3 = LSTMForecaster(input_size=X_train_enh.shape[2],
                         hidden_size=HIDDEN, dropout=DROPOUT, is_volatility=True)
print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")

model3, hist3 = train_model(
    model3, X_train_enh, y_train_vol, X_val_enh, y_val_vol,
    epochs=EPOCHS, batch_size=BATCH, lr=LR, patience=PATIENCE,
    model_name='model3_enh_volatility'
)

# â”€â”€ 9. Generate Out-of-Sample Predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "="*70)
print("  GENERATING OUT-OF-SAMPLE PREDICTIONS")
print("="*70)

pred_ret_m1 = predict(model1, X_test_base)
pred_ret_m2 = predict(model2, X_test_enh)

pred_var_m3 = predict(model3, X_test_enh)

print(f"Predictions: {len(pred_ret_m1)} test observations")

# â”€â”€ 10. Save Forecasts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
lstm_forecasts = pd.DataFrame({
    'Date': test_dates,
    'Realized_Return': y_test_ret,
    'Realized_Vol_Proxy': np.load(os.path.join(DATA_DIR, 'seq_y_test_vol.npy')),
    'LSTM_M1_Mean': pred_ret_m1,
    'LSTM_M2_Mean': pred_ret_m2,
    'LSTM_M3_Var': pred_var_m3
})

lstm_forecasts.to_csv(OUTPUT_CSV, index=False)
print(f"\nLSTM forecasts saved: {OUTPUT_CSV}")
print(f"Shape: {lstm_forecasts.shape}")
print(f"Date range: {lstm_forecasts['Date'].iloc[0]} â†’ {lstm_forecasts['Date'].iloc[-1]}")

# â”€â”€ 11. Training Curves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['Model 1 (Base Return)', 'Model 2 (Enh Return)', 'Model 3 (Enh Volatility)']
for i, (hist, title) in enumerate(zip([hist1, hist2, hist3], titles)):
    axes[i].plot(hist['loss'], label='Train Loss', linewidth=1.2)
    axes[i].plot(hist['val_loss'], label='Val Loss', linewidth=1.2)
    best_epoch = np.argmin(hist['val_loss'])
    axes[i].axvline(best_epoch, color='gray', linestyle='--', alpha=0.5,
                    label=f'Best (epoch {best_epoch+1})')
    axes[i].set_title(title)
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('MSE')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'lstm_training_curves.png'), dpi=300)
plt.close()
print(f"Training curves saved: {PLOTS_DIR}/lstm_training_curves.png")

# â”€â”€ 12. Hyperparameter Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
summary = pd.DataFrame({
    'Model': ['LSTM_M1_Base', 'LSTM_M2_Enhanced', 'LSTM_M3_Volatility'],
    'Target': ['Return', 'Return', 'r^2 (volatility proxy)'],
    'Features': ['Log_Return', 'Log_Return + r^2 + RolVol21 + RolMAE21',
                 'Log_Return + r^2 + RolVol21 + RolMAE21'],
    'Lookback': [60, 60, 60],
    'LSTM_Units': [HIDDEN]*3,
    'Dropout': [DROPOUT]*3,
    'Learning_Rate': [LR]*3,
    'Batch_Size': [BATCH]*3,
    'Max_Epochs': [EPOCHS]*3,
    'Epochs_Trained': [len(h['loss']) for h in [hist1, hist2, hist3]],
    'Best_Val_Loss': [min(h['val_loss']) for h in [hist1, hist2, hist3]],
    'Framework': ['PyTorch']*3,
    'Optimizer': ['Adam']*3,
    'Loss_Function': ['MSE']*3,
    'Seed': [SEED]*3,
})
summary.to_csv(os.path.join(BASE_DIR, 'lstm_hyperparameters.csv'), index=False)
print("\nLSTM Hyperparameter Summary:")
print(summary[['Model', 'Target', 'Epochs_Trained', 'Best_Val_Loss']].to_string(index=False))

print("\nâœ… LSTM modeling complete.")
