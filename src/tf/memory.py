import tensorflow as tf
from tensorflow.keras import Model


def _convolve(w, s):
    assert s.shape[1] == 3
    first_col = tf.slice(w, [0, 0], [-1, 1])
    last_col = tf.slice(w, [0, w.shape[1] - 1], [-1, 1])
    t = tf.concat([last_col, w, first_col], axis=1)
    s = tf.cast(s, dtype='float32')
    c = tf.nn.conv1d(tf.reshape(t, (1, -1, 1)), tf.reshape(s, (-1, 1, 1)), stride=1, padding='VALID')
    c = tf.reshape(c, (-1,))
    return c


class NTMMemory(Model):
    def __init__(self, n_rows: int, n_cols: int):
        super(NTMMemory, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols

        # Initialize memory tensor
        self.prev_mem = None
        self.mem = None

    def reset(self):
        self.mem = tf.ones(shape=(self.n_rows, self.n_cols), dtype='float32') * 1e-6

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

    @staticmethod
    def _interpolate(w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    @staticmethod
    def _shift(wg, s):
        result = _convolve(wg, s)
        return result

    @staticmethod
    def _sharpen(w_hat, gamma):
        w = tf.pow(w_hat, gamma)
        w = w / (tf.math.reduce_sum(w) + 1e-16)
        return w
