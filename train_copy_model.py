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
optimizer = tf.keras.optimizers.Nadam()

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


def train_step(batch):
    losses = []

    with tf.GradientTape() as tape:
        for seq in batch:
            ntm.reset()

            for b in seq:
                ntm(tf.convert_to_tensor([[b]]))
            for b in seq:
                y_true = tf.convert_to_tensor([[b]])
                pred = ntm(tf.convert_to_tensor([[-1]], dtype='float32'))
                loss = loss_object(y_true, pred)
                losses.append(loss)

        loss = tf.reduce_mean(losses)
        gradients = tape.gradient(loss, ntm.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1) for g in gradients]
        optimizer.apply_gradients(zip(gradients, ntm.trainable_variables))


def create_val_set():
    val_set = []
    for _ in range(100):
        length = np.random.randint(1, 21)
        seq = np.random.randint(2, size=length)
        val_set.append(seq)
    return val_set


def eval(val_set, i, min_loss):
    losses = []
    for seq in val_set:
        ntm.reset()

        for b in seq:
            ntm(tf.convert_to_tensor([[b]]))
        for b in seq:
            y_true = tf.convert_to_tensor([[b]])
            pred = ntm(tf.convert_to_tensor([[-1]], dtype='float32'))
            loss = loss_object(y_true, pred)
            losses.append(loss)

    loss = tf.reduce_mean(losses)
    with train_summary_writer.as_default():
        tf.summary.scalar('eval_loss', loss.numpy(), step=i)
    print(i, loss.numpy())

    if loss.numpy() < min_loss:
        min_loss = loss.numpy()
        ntm.save_weights('copy_model/weights', save_format='tf')
    return min_loss


def train():
    min_loss = float('inf')

    val_set = create_val_set()

    for i in range(10000):
        batch = []
        for _ in range(10):
            length = np.random.randint(1, 21)
            seq = np.random.randint(2, size=length)
            batch.append(seq)

        train_step(batch)

        if i % 10 == 0:
            min_loss = eval(val_set, i, min_loss)
            if min_loss < 1e-3:
                break


if __name__ == '__main__':
    train()
