from contextlib import closing
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error
import random
import os

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


closing_prices = pd.read_csv('SP500_Closing_Prices - Copy.csv', header=None).fillna(method='ffill').fillna(method='bfill')
sentiment_scores1 = pd.read_csv('sentiment_repeated_gaza.csv', header=None).fillna(method='ffill').fillna(method='bfill')
herding = pd.read_csv('herding_csad_full.csv', header=None).fillna(method='ffill').fillna(method='bfill')
adjacency_matrix_raw = pd.read_csv('herding_adjacency_matrix.csv', header=None).to_numpy()
num_stocks = 50
closing_prices = closing_prices.iloc[:, :num_stocks]
adjacency_matrix_raw = adjacency_matrix_raw[:num_stocks, :num_stocks]

num_repeats = 50

def repeat_first_column(df, num_repeats):
    df_repeated = df[[0]].copy()
    df_repeated = pd.concat([df_repeated] * num_repeats, axis=1)
    return df_repeated

zeros = pd.DataFrame(np.zeros((90, 1)), columns=[0])
sentiment_scores1 = pd.concat([sentiment_scores1, zeros], ignore_index=True)
sentiment_scores1 = repeat_first_column(sentiment_scores1, num_repeats)
herding = repeat_first_column(herding, num_repeats)

closing_prices = closing_prices.to_numpy()
sentiment_scores1 = sentiment_scores1.to_numpy()
herding = herding.to_numpy()

num_time_steps = closing_prices.shape[0]

train_ratio = 0.7
test_ratio = 0.1
step_size = int(num_time_steps * test_ratio)
initial_train_size = int(num_time_steps * train_ratio)

all_walk_forward_metrics = []

walk_forward_step = 0
start_train = 0

while start_train + initial_train_size + step_size <= num_time_steps:

    walk_forward_step += 1
    print(f"\n================ WALK-FORWARD STEP {walk_forward_step} ================\n")

    end_train = start_train + initial_train_size
    end_test = end_train + step_size

    cp_train = closing_prices[start_train:end_train]
    cp_test = closing_prices[end_train:end_test]

    sent_train = sentiment_scores1[start_train:end_train]
    sent_test = sentiment_scores1[end_train:end_test]

    scaler = MinMaxScaler(feature_range=(0, 1))
    cp_train = scaler.fit_transform(cp_train)
    cp_test = scaler.transform(cp_test)

    train_array = np.stack([cp_train, sent_train], axis=-1)
    test_array = np.stack([cp_test, sent_test], axis=-1)

    train_dataset = create_tf_dataset(train_array, input_sequence_length, forecast_horizon, batch_size)
    test_dataset = create_tf_dataset(
        test_array,
        input_sequence_length,
        forecast_horizon,
        batch_size=test_array.shape[0],
        shuffle=False
    )

    st_gcn = LSTMGC(
        in_feat=in_feat,
        out_feat=out_feat,
        lstm_units=lstm_units,
        input_seq_len=input_sequence_length,
        output_seq_len=forecast_horizon,
        graph_info=graph
    )

    inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
    outputs = st_gcn(inputs)
    model = keras.models.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )

    model.fit(train_dataset, epochs=50, verbose=0)

    X_test, y_test = [], []
    for batch_x, batch_y in test_dataset:
        X_test.append(batch_x.numpy())
        y_test.append(batch_y.numpy())

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    y_pred = model.predict(X_test, verbose=0)

    if y_pred.shape[1] > 1:
        y_pred = y_pred[:, 0:1, :]

    y_true = y_test[:, 0, :]
    y_pred = y_pred[:, 0, :]

    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)

    def mape(y_t, y_p):
        return np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(np.mean((y_true_inv - y_pred_inv) ** 2))
    medae = median_absolute_error(y_true_inv, y_pred_inv)
    mape_val = mape(y_true_inv, y_pred_inv)

    print(f"Step {walk_forward_step} MAE  : {mae:.6f}")
    print(f"Step {walk_forward_step} RMSE : {rmse:.6f}")
    print(f"Step {walk_forward_step} MedAE: {medae:.6f}")
    print(f"Step {walk_forward_step} MAPE : {mape_val:.3f}%")

    all_walk_forward_metrics.append([mae, rmse, medae, mape_val])

    start_train += step_size


all_walk_forward_metrics = np.array(all_walk_forward_metrics)

print("\n================ FINAL WALK-FORWARD RESULT ================\n")
print(f"MAE  : {np.mean(all_walk_forward_metrics[:,0]):.6f}")
print(f"RMSE : {np.mean(all_walk_forward_metrics[:,1]):.6f}")
print(f"MedAE: {np.mean(all_walk_forward_metrics[:,2]):.6f}")
print(f"MAPE : {np.mean(all_walk_forward_metrics[:,3]):.3f}%")
print("============================================================")
