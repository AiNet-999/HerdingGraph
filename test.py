import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, median_absolute_error,
    r2_score, explained_variance_score
)
import matplotlib.pyplot as plt

print("\nRunning evaluation on full test set...")

X_test, y_test = [], []
for batch_x, batch_y in test_dataset:
    X_test.append(batch_x.numpy())
    y_test.append(batch_y.numpy())

X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

y_pred = model.predict(X_test, verbose=1)
print(f"y_pred shape: {y_pred.shape}")

if y_pred.shape[1] > 1:
    y_pred = y_pred[:, 0:1, :]
    print(f"Using first horizon step only → new shape {y_pred.shape}")

y_true = y_test[:, 0, :]
y_pred = y_pred[:, 0, :]

num_stocks = y_true.shape[1]

def mape(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def smape(y_true, y_pred):
    eps = 1e-8
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

def asymmetric_loss(y_true, y_pred, lam=2.0):
    under_mask = y_pred < y_true
    loss = np.abs(y_true - y_pred) * (1 + (lam - 1) * under_mask)
    return np.mean(loss)

def under_over_ratio(y_true, y_pred):
    under = np.sum(y_pred < y_true)
    over = np.sum(y_pred >= y_true)
    total = under + over
    return under / total, over / total

def rmspe(y_true, y_pred):
    eps = 1e-8
    return np.sqrt(np.mean(((y_true - y_pred) / (y_true + eps)) ** 2)) * 100

scaled_metrics = {
    'MAE': [], 'MedAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'SMAPE': [],
    'Bias': [], 'AsymLoss': [], 'UnderRatio': [], 'OverRatio': [],
    'R2': [], 'EVS': []
}

for stock_id in range(num_stocks):
    y_t = y_true[:, stock_id]
    y_p = y_pred[:, stock_id]

    scaled_metrics['MAE'].append(mean_absolute_error(y_t, y_p))
    scaled_metrics['MedAE'].append(median_absolute_error(y_t, y_p))
    mse_val = mean_squared_error(y_t, y_p)
    scaled_metrics['MSE'].append(mse_val)
    scaled_metrics['RMSE'].append(np.sqrt(mse_val))
    scaled_metrics['MAPE'].append(mape(y_t, y_p))
    scaled_metrics['SMAPE'].append(smape(y_t, y_p))
    scaled_metrics['Bias'].append(bias(y_t, y_p))
    scaled_metrics['AsymLoss'].append(asymmetric_loss(y_t, y_p))
    ur, or_ = under_over_ratio(y_t, y_p)
    scaled_metrics['UnderRatio'].append(ur)
    scaled_metrics['OverRatio'].append(or_)
    scaled_metrics['R2'].append(r2_score(y_t, y_p))
    scaled_metrics['EVS'].append(explained_variance_score(y_t, y_p))

print("\nScaled-Space Metrics (0–1 normalized) — Averaged over stocks:")
print("--------------------------------------------------")
print(f"MAE     : {np.mean(scaled_metrics['MAE']):.6f}")
print(f"MedAE   : {np.mean(scaled_metrics['MedAE']):.6f}")
print(f"MSE     : {np.mean(scaled_metrics['MSE']):.6f}")
print(f"RMSE    : {np.mean(scaled_metrics['RMSE']):.6f}")
print(f"MAPE    : {np.mean(scaled_metrics['MAPE']):.3f}%")
print(f"SMAPE   : {np.mean(scaled_metrics['SMAPE']):.3f}%")
print(f"Bias    : {np.mean(scaled_metrics['Bias']):.6f}")
print(f"AsymLoss: {np.mean(scaled_metrics['AsymLoss']):.6f}")
print(f"Underprediction Ratio: {np.mean(scaled_metrics['UnderRatio']):.3f}")
print(f"Overprediction Ratio : {np.mean(scaled_metrics['OverRatio']):.3f}")
print(f"R²      : {np.mean(scaled_metrics['R2']):.6f}")
print(f"EVS     : {np.mean(scaled_metrics['EVS']):.6f}")
print("--------------------------------------------------")

y_true_inv = scaler.inverse_transform(y_true)
y_pred_inv = scaler.inverse_transform(y_pred)

inv_metrics = {
    'MAE': [], 'MedAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'SMAPE': [],
    'Bias': [], 'AsymLoss': [], 'UnderRatio': [], 'OverRatio': [],
    'R2': [], 'EVS': [], 'RMSPE': []
}

for stock_id in range(num_stocks):
    y_t = y_true_inv[:, stock_id]
    y_p = y_pred_inv[:, stock_id]

    inv_metrics['MAE'].append(mean_absolute_error(y_t, y_p))
    inv_metrics['MedAE'].append(median_absolute_error(y_t, y_p))
    mse_val = mean_squared_error(y_t, y_p)
    inv_metrics['MSE'].append(mse_val)
    inv_metrics['RMSE'].append(np.sqrt(mse_val))
    inv_metrics['MAPE'].append(mape(y_t, y_p))
    inv_metrics['SMAPE'].append(smape(y_t, y_p))
    inv_metrics['Bias'].append(bias(y_t, y_p))
    inv_metrics['AsymLoss'].append(asymmetric_loss(y_t, y_p))
    ur, or_ = under_over_ratio(y_t, y_p)
    inv_metrics['UnderRatio'].append(ur)
    inv_metrics['OverRatio'].append(or_)
    inv_metrics['R2'].append(r2_score(y_t, y_p))
    inv_metrics['EVS'].append(explained_variance_score(y_t, y_p))
    inv_metrics['RMSPE'].append(rmspe(y_t, y_p))

print("\nInverse-Scaled (Real Price Space) Metrics — Averaged over stocks:")
print("--------------------------------------------------")
print(f"Model MAE : {np.mean(inv_metrics['MAE']):.5f}")
print(f"Model RMSE: {np.mean(inv_metrics['RMSE']):.5f}")
print(f"MedAE     : {np.mean(inv_metrics['MedAE']):.5f}")
print(f"Bias      : {np.mean(inv_metrics['Bias']):.5f}")
print(f"MAPE      : {np.mean(inv_metrics['MAPE']):.3f}% | SMAPE: {np.mean(inv_metrics['SMAPE']):.3f}%")
print(f"AsymLoss  : {np.mean(inv_metrics['AsymLoss']):.6f}")
print(f"Underpred : {np.mean(inv_metrics['UnderRatio']):.3f} | Overpred: {np.mean(inv_metrics['OverRatio']):.3f}")
print(f"R²        : {np.mean(inv_metrics['R2']):.5f} | EVS: {np.mean(inv_metrics['EVS']):.5f}")
print(f"RMSPE (%) : {np.mean(inv_metrics['RMSPE']):.3f}")
print("--------------------------------------------------")

stock_id = 0
plt.figure(figsize=(10, 4))
plt.plot(y_true_inv[:, stock_id], label="True", linewidth=2)
plt.plot(y_pred_inv[:, stock_id], label="Predicted", linewidth=2)
plt.title(f"Stock {stock_id} - True vs Predicted Prices (H=1 of {forecast_horizon})")
plt.xlabel("Time step")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()