import numpy as np
import tensorflow as tf
from src.tf.ntm import NTM

ntm = NTM()
ntm.mem.size()


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')


def train_step(inputs, i):
    losses = []

    with tf.GradientTape() as tape:
        for bits in inputs:
            n = int(bits.shape[0] / 2)
            ntm.mem.reset()

            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            for b in bits[:n]:
                ntm(tf.convert_to_tensor([[b]]), training=True)
            for (b, delim) in zip(bits[:n], bits[n:]):
                y_true = tf.convert_to_tensor([[b]])
                pred = ntm(tf.convert_to_tensor([[delim]]), training=True)
                loss = loss_object(y_true, pred)
                losses.append(loss)
        loss = tf.reduce_sum(losses)

        print(i, loss.numpy())
        gradients = tape.gradient(loss, ntm.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ntm.trainable_variables))

        train_loss(loss)


for i in range(10000):
    inputs = np.random.randint(2, size=(50, 6))
    inputs[:, 3:] = -1
    train_step(inputs, i)