from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from src.tf.ff_controller import FFController
from src.tf.memory import NTMMemory
from src.tf.read_write_heads import NTMReadHead, NTMWriteHead


class NTM(Model):
    def __init__(self, n_heads=1):
        super(NTM, self).__init__()

        self.controller = FFController()
        self.mem = NTMMemory(100, 8)
        self.read_heads = [NTMReadHead(self.mem, 32) for _ in range(n_heads)]
        self.write_heads = [NTMWriteHead(self.mem, 32) for _ in range(n_heads)]
        self.prev_reads = tf.keras.initializers.GlorotNormal()(shape=(1, 8*n_heads), dtype='float32')

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

