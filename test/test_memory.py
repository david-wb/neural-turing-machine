import pytest
import numpy as np
from src.tf.memory import NTMMemory
import tensorflow as tf


def test_memory():
    mem = NTMMemory(10, 3)
    mem.reset()


def test_read():
    mem = NTMMemory(10, 3)
    mem.reset()

    w = np.zeros(shape=(10,), dtype='float32')
    w[2] = 1
    w = tf.convert_to_tensor(w)
    r = mem.read(w)
    a = r.numpy()
    b = mem.mem.numpy()[2, :]

    assert np.array_equal(a, b)


def test_write():
    mem = NTMMemory(10, 3)
    mem.reset()

    w = np.zeros(shape=(10,), dtype='float32')
    w[2] = 1
    w = tf.convert_to_tensor(w)
    e = tf.ones(shape=(3,))
    a = tf.convert_to_tensor([0.5, 1, 0.3])
    mem.write(w, e, a)

    assert np.array_equal(mem.mem.numpy()[2, :], a.numpy())


def test_address():
    mem = NTMMemory(10, 3)
    mem.reset()

    row_5 = mem.mem.numpy()[4, :]

    w = np.zeros(shape=(10,), dtype='float32')
    w[1] = 1
    w = tf.convert_to_tensor(w)

    next_w = mem.address(k=tf.convert_to_tensor(row_5), beta=1, g=1, s=tf.constant([0, 1, 0]), gamma=10, w_prev=w)

    assert np.argmax(next_w.numpy()) == 4
    assert np.sum(next_w.numpy()) == 1
