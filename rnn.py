import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential()

model.add(
    layers.Bidirectional(layers.LSTM(32, # dimensionality of the output space
                                     activation="tanh",
                                     # recurrent_dropout=0.01, # remove this to use CuDNN
                                     return_sequences=True),
                         input_shape= None,
                         merge_mode="concat"))  # (5, 10)) #
)
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(10))

model.summary()
model.compile(optimizer ="adam",
              loss=keras.losses.CategoricalCrossentropy,
              metrics=["accuracy", tf.keras.metrics.CategoricalAccuracy()])

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=1)

model.save("model.h5")
model.summary()

# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
# https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/

