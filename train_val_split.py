from contextlib import closing
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import random
import os
import matplotlib.pyplot as plt

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

closing_prices = pd.read_csv('/kaggle/input/mynewdataset/Sentiment/SP500_Closing_Prices - Copy.csv', header=None).fillna(method='ffill').fillna(method='bfill')
sentiment_scores1 = pd.read_csv('/kaggle/input/mynewdataset/Sentiment/sentiment_repeated_gaza.csv', header=None).fillna(method='ffill').fillna(method='bfill')
herding = pd.read_csv('/kaggle/input/datasets/maryambukhari/revision-herd/herding_csad_full.csv', header=None).fillna(method='ffill').fillna(method='bfill')
adjacency_matrix_raw = pd.read_csv('/kaggle/input/datasets/maryambukhari/final-herding-revision-data/herding_adjacency_matrix.csv', header=None).to_numpy()

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

train_end = int(num_time_steps * 0.9)

cp_train = closing_prices[:train_end]
cp_test = closing_prices[train_end:]

herd_train = herding[:train_end]
herd_test = herding[train_end:]

sent_train = sentiment_scores1[:train_end]
sent_test = sentiment_scores1[train_end:]

scaler = MinMaxScaler(feature_range=(0, 1))
cp_train = scaler.fit_transform(cp_train)
cp_test = scaler.transform(cp_test)

train_array = np.stack([cp_train, sent_train], axis=-1)
test_array = np.stack([cp_test, sent_test], axis=-1)


val_split = int(len(train_array) * 0.9)

train_array_final = train_array[:val_split]
val_array = train_array[val_split:]


def create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size, shuffle=True):
    inputs = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[:-forecast_horizon],
        None,
        sequence_length=input_sequence_length,
        batch_size=batch_size,
        shuffle=False,
    )

    targets = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[input_sequence_length + forecast_horizon - 1:, :, 0],
        None,
        sequence_length=1,
        batch_size=batch_size,
        shuffle=False,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()

in_feat = 2
out_feat = 3
lstm_units = 64
input_sequence_length = 3
forecast_horizon = 1
batch_size = 8
learning_rate = 0.001
epsilon = 0.5


train_dataset = create_tf_dataset(train_array_final, input_sequence_length, forecast_horizon, batch_size)
val_dataset = create_tf_dataset(val_array, input_sequence_length, forecast_horizon, batch_size, shuffle=False)
test_dataset = create_tf_dataset(test_array, input_sequence_length, forecast_horizon, batch_size=test_array.shape[0], shuffle=False)


class GraphInfo:
    def __init__(self, edges, num_nodes):
        self.edges = edges
        self.num_nodes = num_nodes

def compute_adjacency_matrix(route_distances, epsilon):
    return np.where(route_distances > epsilon, 1, 0)

adjacency_matrix = compute_adjacency_matrix(adjacency_matrix_raw, epsilon)
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)

graph = GraphInfo(edges=(node_indices.tolist(), neighbor_indices.tolist()), num_nodes=adjacency_matrix.shape[0])


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


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    verbose=1
)


test_loss = model.evaluate(test_dataset)

print("\nFinal Test Loss:", test_loss)


plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()
