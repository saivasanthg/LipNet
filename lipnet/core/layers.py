import tensorflow as tf
from tensorflow.keras.layers import Layer

class CTC(Layer):
    def call(self, inputs):
        y_pred, labels, input_length, label_length = inputs

        # CTC loss expects time major format
        y_pred = tf.transpose(y_pred, [1, 0, 2])

        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
