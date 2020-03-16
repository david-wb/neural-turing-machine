"""An NTM's memory implementation."""
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.shape[1] == 3
    first_col = tf.slice(w, [0, 0], [-1, 1])
    last_col = tf.slice(w, [0, w.shape[1] - 1], [-1, 1])
    t = tf.concat([last_col, w, first_col], axis=1)
    s = tf.cast(s, dtype='float32')
    c = tf.nn.conv1d(tf.reshape(t, (1, -1, 1)), tf.reshape(s, (-1, 1, 1)), stride=1, padding='VALID')
    c = tf.reshape(c, (-1,))
    return c


class NTMMemory(Model):
    """Memory bank for NTM."""
    def __init__(self, n_rows: int, n_cols: int):
        """Initialize the NTM Memory matrix.
        The memory's dimensions are (n_rows x n_cols).
        Each batch has it's own memory matrix.
        :param n_rows: Number of rows in the memory.
        :param n_cols: Number of columns in the memory.
        """
        super(NTMMemory, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols

        # Initialize memory tensor
        self.prev_mem = None
        #self.mem = tf.zeros(shape=(n_rows, n_cols), dtype='float32')
        self.mem = tf.keras.initializers.GlorotNormal()(shape=(n_rows, n_cols), dtype='float32') * 1e-8

    def reset(self):
        # self.mem = tf.zeros(shape=(self.n_rows, self.n_cols), dtype='float32')
        self.mem = tf.keras.initializers.GlorotNormal()(shape=(self.n_rows, self.n_cols), dtype='float32')

    def size(self):
        return self.n_rows, self.n_cols

    def read(self, weights: tf.Tensor):
        return tf.linalg.matvec(tf.transpose(self.mem), weights)

    def write(self, w, e, a):
        self.prev_mem = self.mem
        erase = tf.matmul(tf.transpose(w), e)
        add = tf.matmul(tf.transpose(w), a)
        self.mem = self.prev_mem * (1 - erase) + add

    def address(self, k, beta, g, s, gamma, w_prev):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param k: The key vector.
        :param beta: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param gamma: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        wc = self._similarity(k, beta)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        w_hat = self._shift(wg, s)
        w = self._sharpen(w_hat, gamma)

        return w

    def _similarity(self, k, beta):
        cos_sim = 1 - tf.keras.losses.cosine_similarity(self.mem + 1e-16, k + 1e-16)
        w = tf.nn.softmax(beta * cos_sim, axis=-1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = tf.zeros(wg.shape)
        result = _convolve(wg, s)
        return result

    def _sharpen(self, w_hat, gamma):
        w = tf.pow(w_hat, gamma)
        w = w / (tf.math.reduce_sum(w) + 1e-16)
        return w