"""An NTM's memory implementation."""
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.shape(0) == 3
    t = tf.concat([w[-1:], w, w[:1]], axis=0)
    c = tf.nn.conv1d(tf.reshape(t, (1, 1, -1)), tf.reshape(s, (-1, 1, 1)))
    c = tf.reshape(c, (-1,))
    return c


class NTMMemory(Model):
    """Memory bank for NTM."""
    def __init__(self, n_rows: int, n_cols: int):
        """Initialize the NTM Memory matrix.
        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.
        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols

        # Initialize memory tensor
        self.prev_mem = None
        self.mem: tf.Tensor = tf.zeros((n_rows, n_cols))

        # Initialize memory bias tensor
        stdev = 1 / (np.sqrt(n_rows + n_cols))
        self.mem_bias: tf.Tensor = tf.random.uniform((n_rows, n_cols), -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.mem.assign(self.mem_bias)

    def size(self):
        return self.N, self.M

    def read(self, weights: tf.Tensor):
        """Read from memory (according to section 3.1)."""
        return tf.linalg.matvec(tf.transpose(self.mem), weights)

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.mem
        erase = tf.matmul(tf.expand_dims(w, -1), tf.expand_dims(e, 0))
        add = tf.matmul(tf.expand_dims(w, -1), tf.expand_dims(a, 0))
        self.mem = self.prev_mem * (1 - erase) + add

    def address(self, k, β, g, s, γ, w_prev):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        wc = self._similarity(k, β)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)

        return w

    def _similarity(self, k, β):
        cos_sim = tf.keras.losses.cosine_similarity(self.memory + 1e-16, k + 1e-16)
        w = tf.nn.softmax(β * cos_sim, axis=-1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = tf.zeros(wg.shape)
        result= _convolve(wg, s)
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = w / (tf.math.reduce_sum(w) + 1e-16)
        return w