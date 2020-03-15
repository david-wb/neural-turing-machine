from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from src.tf.memory import NTMMemory


class FFController(Model):
    def __init__(self, output_size=100):
        super(FFController, self).__init__()
        self.d1 = Dense(output_size, activation='relu')
        self.fc_external_out = Dense(1)
        self.fc_internal_out = Dense(output_size, activation='relu')

    def call(self, inputs):
        x = tf.concat(inputs, axis=-1)
        x = self.d1(x)
        external_out = self.fc_external_out(x)
        internal_out = self.fc_internal_out(x)
        return internal_out, external_out


