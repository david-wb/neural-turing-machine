"""NTM Read and Write Heads."""
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from src.tf.memory import NTMMemory


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class NTMHeadBase(Model):
    """An NTM Read/Write Head."""

    def __init__(self, memory, controller_size):
        """Initilize the read/write head.
        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super(NTMHeadBase, self).__init__()

        self.mem: NTMMemory = memory
        self.n_rows, self.n_cols = memory.size()
        self.controller_size = controller_size
        self.w = None

        init_w = tf.initializers.glorot_normal()(shape=(1, self.n_rows))

        init_w = np.zeros((1, self.n_rows), dtype='float32')
        init_w[0, 0] = 100
        self.init_w = tf.Variable(init_w, name='init_w')
        self.reset()

    def reset(self):
        self.w = tf.nn.softmax(self.init_w)

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, beta, g, s, gamma, w_prev):
        # Handle Activations
        beta = tf.nn.softplus(beta)
        g = tf.nn.sigmoid(g)
        s = tf.nn.softmax(s)
        gamma = 1 + tf.nn.softplus(gamma)
        w = self.mem.address(k, beta, g, s, gamma, w_prev)
        return w


class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_output_size):
        super(NTMReadHead, self).__init__(memory, controller_output_size)

        # The read head should output the variables k, beta, g, s, and gamma described in the paper.
        # k is the key vector with length equal to the number of columns in the memory matrix.
        # beta, g, and gamma are scalars. s is the shift vector which usually has length 3.
        # So the output size should be n_cols + 6.
        self.output_size = self.n_cols + 6
        self.fc_read = Dense(self.output_size, input_shape=(controller_output_size,))

    def is_read_head(self):
        return True

    def call(self, x):
        x = self.fc_read(x)
        k, beta, g, s, gamma = tf.split(x, [self.n_cols, 1, 1, 3, 1], axis=-1)

        # Read from memory
        w = self._address_memory(k, beta, g, s, gamma, self.w)
        r = self.mem.read(w)

        self.w = w
        return r


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_output_size):
        super(NTMWriteHead, self).__init__(memory, controller_output_size)

        # The read head should output the variables k, beta, g, s, gamma, e, and a described in the paper.
        # k is the key vector with length equal to the number of columns in the memory matrix.
        # e and a are the erase and add vectors which also have same length as the key vector.
        # beta, g, and gamma are scalars. s is the shift vector which usually has length 3.
        # So the output size should be n_cols * 3 + 6.
        self.output_size = self.n_cols * 3 + 6
        self.fc_write = Dense(self.output_size, input_shape=(controller_output_size,))

    def is_read_head(self):
        return False

    def call(self, x):
        x = self.fc_write(x)
        k, beta, g, s, gamma, e, a = tf.split(x, [self.n_cols, 1, 1, 3, 1, self.n_cols, self.n_cols], axis=-1)

        # e should be in [0, 1]
        e = tf.nn.sigmoid(e)

        # Write to memory
        w = self._address_memory(k, beta, g, s, gamma, self.w)
        self.mem.write(w, e, a)
        self.w = w
        return w
