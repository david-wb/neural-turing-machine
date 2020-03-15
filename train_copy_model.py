import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from src.tf.ntm import NTM

ntm = NTM()
ntm.reset()

if os.path.exists('./copy_model'):
    print('loading weights')
    ntm.load_weights('copy_model/weights')

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


def train_step(batch, i, min_loss):
    losses = []

    with tf.GradientTape() as tape:
        for seq in batch:
            ntm.reset()

            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            for b in seq:
                ntm(tf.convert_to_tensor([[b]]), training=False)
            for b in seq:
                y_true = tf.convert_to_tensor([[b]])
                pred = ntm(tf.convert_to_tensor([[-1]]), training=True)
                loss = loss_object(y_true, pred)
                losses.append(loss)
        loss = tf.reduce_mean(losses)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss.numpy(), step=i)
        print(i, loss.numpy())
        gradients = tape.gradient(loss, ntm.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ntm.trainable_variables))

        train_loss(loss)

    if loss.numpy() < min_loss:
        min_loss = loss.numpy()
        ntm.save_weights('copy_model/weights', save_format='tf')
    return min_loss


def train():
    min_loss = float('inf')

    for i in range(300):
        batch = []
        for _ in range(50):
            length = np.random.randint(1, 20)
            example = np.random.randint(2, size=length)
            example[length:] = -1
            batch.append(example)
        min_loss = train_step(batch, i, min_loss)


if __name__ == '__main__':
    train()
