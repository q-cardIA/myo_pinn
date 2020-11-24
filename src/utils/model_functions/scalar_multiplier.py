# import necessary modules
import tensorflow as tf
from tensorflow.keras.layers import Layer


class ScalarMultiply(Layer):
    def __init__(self, value, **kwargs):
        super(ScalarMultiply, self).__init__(**kwargs)
        self.scalar = tf.Variable(value, dtype=tf.float32)

    def call(self, x):
        return tf.scalar_mul(self.scalar, x)
