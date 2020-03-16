from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from src.tf.memory import NTMMemory


class FFController(Model):
    def __init__(self, output_size=100, external_output_size=1):
        super(FFController, self).__init__()
        self.d1 = Dense(output_size, activation='relu')
        self.fc_external_out = Dense(external_output_size)
        self.fc_internal_out = Dense(output_size, activation='relu')

    def call(self, inputs):
        x, prev_read = inputs
        x = tf.reshape(x, [1, -1])
        x = tf.concat([x, prev_read], axis=-1)
        x = self.d1(x)
        external_out = self.fc_external_out(x)
        internal_out = self.fc_internal_out(x)
        return internal_out, external_out


