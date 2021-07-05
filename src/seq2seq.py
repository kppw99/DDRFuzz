#!/usr/bin/python3

import os
import time
import tensorflow as tf
from util import *
from models import *


class EncoderS(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(EncoderS, self).__init__()
        self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(units, return_state=True)

    def call(self, inputs):
        x = self.emb(inputs)
        mask = self.emb.compute_mask(inputs)
        _, h, c = self.lstm(x, mask=mask)
        return h, c


class DecoderS(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(DecoderS, self).__init__()
        self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(OUT_DIM, activation='softmax')

    def call(self, inputs):
        x, h, c = inputs
        x1 = self.emb(x)
        mask = self.emb.compute_mask(x)
        x, h, c = self.lstm(x1, initial_state=[h, c], mask=mask)
        return self.dense(x), h, c


class Seq2seq(tf.keras.Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, embedding_dim, units, sos, eos, maxlen):
        super(Seq2seq, self).__init__()
        self.enc = EncoderS(enc_vocab_size, embedding_dim, units)
        self.dec = DecoderS(dec_vocab_size, embedding_dim, units)
        self.sos = sos
        self.eos = eos
        self.maxlen = maxlen

    def call(self, inputs, training=False):
        if training is True:
            x, y = inputs
            h, c = self.enc(x)
            y, _, _ = self.dec((y, h, c))

            return y
        else:
            x = inputs
            h, c = self.enc(x)
            y = tf.convert_to_tensor(self.sos)
            y = tf.reshape(y, (1, 1))
            seq = tf.TensorArray(tf.int32, self.maxlen)

            for idx in tf.range(self.maxlen):
                y, h, c = self.dec([y, h, c])
                y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
                y = tf.reshape(y, (1, 1))
                seq = seq.write(idx, y)
                if y == self.eos: break

            return tf.reshape(seq.stack(), (1, self.maxlen))


if __name__=='__main__':
    MAXLEN = 1000
    EMBEDDING_DIM = 64
    LSTM_DIM = 256
    BATCH_SIZE = 8
    EPOCHS = 30
    TEST_RATIO = 0.0

    input_tensor, target_tensor = load_dataset('../seq2seq/init_dataset/PNG/path', pad_maxlen=MAXLEN)
    train_ds, test_ds = split_tensor(input_tensor, target_tensor, batch_size=BATCH_SIZE, test_ratio=TEST_RATIO)

    model = Seq2seq(ENC_VOCAB_SIZE, DEC_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, units=LSTM_DIM,
                    sos=SOS, eos=EOS, maxlen=MAXLEN)
    model = train_seq2seq_model(model, train_ds, epochs=EPOCHS, early_stop_patience=5)

    optimizer = tf.keras.optimizers.Adamax()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    dir_path = './saved_model/s2s'
    if (os.path.isdir(dir_path) == False):
        os.mkdir(dir_path)

    filetime = time.strftime("_%Y%m%d-%H%M%S")
    filename = 'simple_s2s_model' + filetime + '.ckpt'
    fullname = os.path.join(dir_path, filename)
    checkpoint.save(fullname)
