from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from src.tf.ff_controller import FFController
from src.tf.memory import NTMMemory
from src.tf.read_write_heads import NTMReadHead, NTMWriteHead


class NTM(Model):
    def __init__(self, n_heads=1, memory_dim=16, memory_size=128, external_output_size=1):
        super(NTM, self).__init__()

        self.memory_dim = memory_dim
        self.n_heads = n_heads
        self.controller = FFController(output_size=100, external_output_size=external_output_size)
        self.mem = NTMMemory(memory_size, memory_dim)
        self.read_heads = [NTMReadHead(self.mem, 100) for _ in range(n_heads)]
        self.write_heads = [NTMWriteHead(self.mem, 100) for _ in range(n_heads)]

        self.prev_reads = None

        init_reads = tf.keras.initializers.GlorotNormal()(shape=(1, memory_dim * n_heads), dtype='float32')
        self.init_reads = tf.Variable(init_reads)

    def reset(self):
        self.mem.reset()
        for rh in self.read_heads:
            rh.reset()
        for wh in self.write_heads:
            wh.reset()
        self.prev_reads = self.init_reads

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype='float32')
        cinternal_out, cexternal_out = self.controller([inputs, self.prev_reads])

        reads = []
        for rh in self.read_heads:
            r = rh(cinternal_out)
            reads.append(r)
        for wh in self.write_heads:
            wh(cinternal_out)

        prev_reads = tf.concat(reads, axis=-1)
        self.prev_reads = prev_reads
        return cexternal_out

