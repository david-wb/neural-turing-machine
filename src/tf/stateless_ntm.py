from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from src.tf.memory import NTMMemory
from src.tf.read_write_heads import NTMReadHead, NTMWriteHead
import numpy as np

from src.tf.stateless_memory import StatelessNTMMemory
from src.tf.stateless_read_write_heads import StatelessNTMReadHead, StatelessNTMWriteHead


class StatelessNTM(Model):
    def __init__(self, n_heads=1, memory_dim=20, memory_size=100, external_output_size=1):
        super(StatelessNTM, self).__init__()

        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.n_heads = n_heads
        self.mem = StatelessNTMMemory(tf.ones(shape=(memory_size, memory_dim), dtype='float32') * 1e-8)
        self.read_head = StatelessNTMReadHead(self.mem, 200)
        self.write_head = StatelessNTMWriteHead(self.mem, 200)

        self.reads_bias = tf.Variable(tf.zeros(shape=(1, memory_dim), dtype='float32'))
        self.fc1 = Dense(200, activation='relu')
        self.fc_out1 = Dense(100, activation='relu')
        self.fc_external_out = Dense(external_output_size)
        self.init_read = tf.Variable(tf.ones(shape=(1, self.memory_dim), dtype='float32') * 1e-8)

        w_init = np.zeros((1, self.memory_size), dtype='float32')
        w_init[0, 0] = 100
        self.init_read_w = tf.Variable(w_init, name='init_read_w')
        self.init_write_w = tf.Variable(w_init, name='init_write_w')

    def get_start_state(self):

        return {
            'M': tf.ones(shape=(self.memory_size, self.memory_dim), dtype='float32') * 1e-8,
            'read_w': tf.nn.softmax(self.init_read_w),
            'write_w': tf.nn.softmax(self.init_write_w),
            'read_prev': self.init_read,
        }

    @tf.function
    def call(self, x, prev_state):
        self.mem.update(prev_state['M'])

        x = tf.cast(x, dtype='float32')
        x = tf.reshape(x, [1, -1])
        x = tf.concat([x, prev_state['read_prev'] + self.reads_bias], axis=-1)
        x = self.fc1(x)

        read, read_w = self.read_head(x, prev_state['read_w'])
        write, write_w = self.write_head(x, prev_state['write_w'])

        x = tf.concat([x, read], axis=-1)
        x = self.fc_out1(x)

        out = self.fc_external_out(x)
        state = {
            'M': self.mem.mem,
            'read_w': read_w,
            'write_w': write_w,
            'read_prev': read
        }

        return out, state

