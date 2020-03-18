import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from src.tf.ntm import NTM

ntm = NTM(external_output_size=18)
ntm.reset()

if os.path.exists('./assoc_model'):
    print('loading weights')
    ntm.load_weights('assoc_model/weights')

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Nadam()

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


def train_step(batch):
    losses = []

    with tf.GradientTape() as tape:
        for seq in batch:
            ntm.reset()

            for item in seq:
                ntm(tf.convert_to_tensor(item))

            query_i = np.random.randint(len(seq) - 2)
            query = seq[query_i]
            y_true = seq[query_i + 1]

            pred = ntm(query)
            pred = tf.reshape(pred, shape=(6, 3))
            loss = loss_object(y_true, pred)
            losses.append(loss)

        loss = tf.reduce_mean(losses)
        gradients = tape.gradient(loss, ntm.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ntm.trainable_variables))


def eval(val_set, i, min_loss):
    losses = []

    for seq in val_set:
        ntm.reset()

        for item in seq:
            ntm(tf.convert_to_tensor(item))

        query_i = np.random.randint(len(seq) - 2)
        query = seq[query_i]
        y_true = seq[query_i + 1]

        pred = ntm(query)
        pred = tf.reshape(pred, shape=(6, 3))
        loss = loss_object(y_true, pred)
        losses.append(loss)

    loss = tf.reduce_mean(losses)
    with train_summary_writer.as_default():
        tf.summary.scalar('eval_loss', loss.numpy(), step=i)
    print(i, loss.numpy())

    if loss.numpy() < min_loss:
        min_loss = loss.numpy()
        ntm.save_weights('assoc_model/weights', save_format='tf')
    return min_loss


def get_batch(size, max_len=6):
    batch = []
    for _ in range(size):
        length = np.random.randint(2, max_len + 1)
        seq = [np.random.randint(2, size=(6, 3)) for _ in range(length)]
        seq.append(np.ones(shape=(6, 3)) * -1)
        batch.append(seq)
    return batch


def train():
    min_loss = float('inf')

    val_set = get_batch(100)

    for i in range(100000):
        batch = get_batch(10)
        train_step(batch)

        if i % 10 == 0:
            min_loss = eval(val_set, i, min_loss)

            if min_loss < 1e-3:
                break


if __name__ == '__main__':
    train()
