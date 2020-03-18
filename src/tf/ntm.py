from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from src.tf.memory import NTMMemory
from src.tf.read_write_heads import NTMReadHead, NTMWriteHead


class NTM(Model):
    def __init__(self, n_heads=1, memory_dim=20, memory_size=100, external_output_size=1):
        super(NTM, self).__init__()

        self.memory_dim = memory_dim
        self.n_heads = n_heads
        self.mem = NTMMemory(memory_size, memory_dim)
        self.read_heads = [NTMReadHead(self.mem, 200) for _ in range(n_heads)]
        self.write_heads = [NTMWriteHead(self.mem, 200) for _ in range(n_heads)]

        self.prev_reads = None

        self.init_reads = tf.Variable(tf.zeros(shape=(1, memory_dim * n_heads), dtype='float32'), name='init_reads')
        self.fc1 = Dense(200, activation='relu')
        self.fc_external_out = Dense(external_output_size)

    def reset(self):
        self.mem.reset()
        for rh in self.read_heads:
            rh.reset()
        for wh in self.write_heads:
            wh.reset()
        self.prev_reads = self.init_reads

    def call(self, inputs):
        x = tf.cast(inputs, dtype='float32')
        x = tf.reshape(x, [1, -1])
        x = tf.concat([x, self.prev_reads], axis=-1)
        x = self.fc1(x)

        reads = []
        for rh in self.read_heads:
            r = rh(x)
            reads.append(r)
        for wh in self.write_heads:
            wh(x)

        reads = tf.concat(reads, axis=-1)

        x = tf.concat([x, reads], axis=-1)
        out = self.fc_external_out(x)
        self.prev_reads = reads
        return out

