from contextlib import closing
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import random
import os

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
closing_prices = pd.read_csv('SP500_Closing_Prices - Copy.csv', header=None).fillna(method='ffill').fillna(method='bfill')
sentiment_scores1 = pd.read_csv('sentiment_gaza.csv', header=None).fillna(method='ffill').fillna(method='bfill')
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

train_end = int(num_time_steps * 0.9)

#approx 30 Aug, 2023 to 30 Aug, 2024 #test period

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

def create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size, shuffle=True, multi_horizon=False):
    inputs = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[:-forecast_horizon],
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )
    target_offset = input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = tf.keras.preprocessing.timeseries_dataset_from_array(
        data_array[target_offset:, :, 0],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )
    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)
    return dataset.prefetch(16).cache()

in_feat = 2
out_feat = 3
lstm_units = 64
input_sequence_length = 5
forecast_horizon = 1
batch_size = 8
learning_rate = 0.001
epsilon = 0.5

class GraphInfo:
    def __init__(self, edges, num_nodes):
        self.edges = edges
        self.num_nodes = num_nodes

def compute_adjacency_matrix(route_distances, epsilon):
    return np.where(route_distances > epsilon, 1, 0)

adjacency_matrix = compute_adjacency_matrix(adjacency_matrix_raw, epsilon)
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)

graph = GraphInfo(edges=(node_indices.tolist(), neighbor_indices.tolist()), num_nodes=adjacency_matrix.shape[0])
print(f"Total number of edges in the graph: {len(node_indices)}")


class GraphConv1(layers.Layer):
    def __init__(self, in_feat, out_feat, graph_info, aggregation_type="mean", combination_type="concat", activation=None, **kwargs):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.activation = layers.Activation(activation)
        src_idx = tf.convert_to_tensor(graph_info.edges[0], dtype=tf.int32)
        tgt_idx = tf.convert_to_tensor(graph_info.edges[1], dtype=tf.int32)
        self.edge_source = src_idx
        self.edge_target = tgt_idx
        self.num_nodes = int(graph_info.num_nodes)
        w_init = keras.initializers.glorot_uniform()
        self.weight = tf.Variable(initial_value=w_init(shape=(in_feat, out_feat), dtype="float32"), trainable=True)

    def aggregate(self, neighbour_representations):
        if self.aggregation_type == "sum":
            agg = tf.math.unsorted_segment_sum(neighbour_representations, self.edge_source, num_segments=self.num_nodes)
        elif self.aggregation_type == "mean":
            agg = tf.math.unsorted_segment_mean(neighbour_representations, self.edge_source, num_segments=self.num_nodes)
        elif self.aggregation_type == "max":
            agg = tf.math.unsorted_segment_max(neighbour_representations, self.edge_source, num_segments=self.num_nodes)
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")
        return agg

    def compute_nodes_representation(self, features):
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features):
        neighbour_representations = tf.gather(features, self.edge_target)
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation, aggregated_messages):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}")
        return self.activation(h)

    def call(self, features):
        nodes_repr = self.compute_nodes_representation(features)
        agg_msgs = self.compute_aggregated_messages(features)
        out = self.update(nodes_repr, agg_msgs)
        return out


class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = None
        self.b = None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="att_weight"
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name="att_bias"
        )

    def call(self, x):

        e = tf.tanh(tf.matmul(x, self.W) + self.b)  # (batch, time, 1)


        a = tf.nn.softmax(e, axis=1)  # (batch, time, 1)

        context = tf.reduce_sum(a * x, axis=1)  # (batch, features)

        return context


class LSTMGC(layers.Layer):
    def __init__(self, in_feat, out_feat, lstm_units,
                 input_seq_len, output_seq_len,
                 graph_info, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.lstm_units = lstm_units
        self.input_seq_len = int(input_seq_len)
        self.output_seq_len = int(output_seq_len)
        self.graph_info = graph_info


        self.graph_conv = GraphConv1(
            in_feat, out_feat, graph_info,
            aggregation_type="mean",
            combination_type="concat",
            activation=None
        )

  
        #self.lstm = layers.LSTM(64, return_sequences=True)
         self.lstm = layers.Bidirectional(
    layers.LSTM(64, return_sequences=True)
)
        #self.lstm1 = layers.LSTM(32, return_sequences=True)


        self.temporal_attention = TemporalAttention()

        self.dropout = layers.Dropout(dropout_rate)


        self.dense = layers.Dense(self.output_seq_len)

    def call(self, inputs, training=False):

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        num_nodes = tf.shape(inputs)[2]

       
        folded = tf.reshape(inputs, (batch_size * seq_len, num_nodes, self.in_feat))
        folded_t = tf.transpose(folded, [1, 0, 2])

        gcn_out_t = self.graph_conv(folded_t)
        final_out_feat = tf.shape(gcn_out_t)[-1]

        gcn_out_folded = tf.transpose(gcn_out_t, [1, 0, 2])
        gcn_out = tf.reshape(
            gcn_out_folded,
            (batch_size, seq_len, num_nodes, final_out_feat)
        )

        gcn_out = tf.transpose(gcn_out, [0, 2, 1, 3])

   
        gcn_for_lstm = tf.reshape(
            gcn_out,
            (batch_size * num_nodes, seq_len, final_out_feat)
        )

        lstm_out = self.lstm(gcn_for_lstm, training=training)
        #lstm_out = self.lstm1(lstm_out, training=training)

        context_vector = self.temporal_attention(lstm_out)

        context_vector = self.dropout(context_vector, training=training)

        dense_out = self.dense(context_vector)

        dense_out = tf.reshape(
            dense_out,
            (batch_size, num_nodes, self.output_seq_len)
        )

        return tf.transpose(dense_out, [0, 2, 1])
train_dataset = create_tf_dataset(train_array, input_sequence_length, forecast_horizon, batch_size)
test_dataset = create_tf_dataset(
    test_array,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_array.shape[0],
    shuffle=False
)
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
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
history = model.fit(train_dataset, epochs=100)

train_loss = history.history['loss'][-1]
test_loss = model.evaluate(test_dataset)

print("\nFinal Training Loss:", train_loss)
print("Final Test Loss:", test_loss)