import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_cnn_bilstm(num_timesteps, num_classes,
                     conv_filters=[32, 64, 128], kernel_size=3,
                     lstm_units=[180], dropout=0.3, l2=1e-4):
    """
    CNN -> BiLSTM with conv stack [32,64,128] and LSTM units default 180 (paper-like).
    """
    inputs = layers.Input(shape=(num_timesteps,), name="input")
    x = layers.Reshape((num_timesteps, 1))(inputs)

    for f in conv_filters:
        x = layers.Conv1D(filters=f, kernel_size=kernel_size, padding='same',
                          activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

    for i, units in enumerate(lstm_units):
        return_seq = True if i < len(lstm_units) - 1 else False
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=return_seq))(x)

    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="cnn_bilstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
