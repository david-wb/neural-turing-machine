import pytest
import numpy as np
from src.tf.memory import NTMMemory
import tensorflow as tf

from src.tf.ntm import NTM


def test_ntm():
    ntm = NTM()
    input = tf.convert_to_tensor([[0]])

    out = ntm(input)
    print(out.numpy()[0][0])
