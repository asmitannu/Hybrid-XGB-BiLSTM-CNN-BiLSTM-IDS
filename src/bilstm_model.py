# src/bilstm_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_bilstm(num_timesteps, num_classes, lstm_units=[128, 64], dropout=0.3, l2=1e-4):
    """
    Build a BiLSTM-only model that accepts num_timesteps-length feature vector
    treated as a sequence (reshape to (num_timesteps,1)).
    """
    inputs = layers.Input(shape=(num_timesteps,), name="input")
    x = layers.Reshape((num_timesteps, 1))(inputs)

    for i, units in enumerate(lstm_units):
        return_seq = True if i < len(lstm_units) - 1 else False
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=return_seq,
                                             kernel_regularizer=regularizers.l2(l2)))(x)

    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="bilstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
