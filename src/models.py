#!/usr/bin/python3

import time
import tensorflow as tf
from util import *
from tensorflow.keras.callbacks import EarlyStopping


class Early_Stopping(tf.keras.Model):
    def __init__(self, patience=0, verbose=0):
        super(Early_Stopping, self).__init__()
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def call(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'[*] Training process is stopped early ...')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


def train_seq2seq_model(model, train_ds, epochs, early_stop_patience=None):
    @tf.function  # Implement training loop
    def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
        output_labels = labels[:, 1:]
        shifted_labels = labels[:, :-1]
        with tf.GradientTape() as tape:
            predictions = model([inputs, shifted_labels], training=True)
            loss = loss_object(output_labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(output_labels, predictions)

    # definition of loss function and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adamax()

    # definition of measurement matrix
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    if early_stop_patience is not None:
        early_stopping = Early_Stopping(patience=early_stop_patience, verbose=True)

    for epoch in range(epochs):
        start = time.perf_counter()
        loop_cnt = 0
        for seqs, labels in train_ds:
            train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)
            loop_cnt += 1
        end = time.perf_counter()
        template = '[Epoch {0:}/{1:}] - {2:0.1f}s {3:0.1f}s/step - loss: {4:0.4f} - accuracy: {5:0.4f}'
        print(template.format(epoch + 1, epochs, (end - start), ((end - start) / loop_cnt),
                              train_loss.result(), train_accuracy.result()))

        if early_stop_patience is not None:
            if early_stopping(train_loss.result()):
                break

    return model


def test_seq2seq_model(model, test_ds, verbose=False, save=False):
    @tf.function  # Implement Inference
    def test_step(model, inputs):
        return model(inputs, training=False)

    for idx, (test_seq, test_labels) in enumerate(test_ds):
        print('[*]', idx)
        prediction = test_step(model, test_seq)

        if verbose is True:
            print('====================')
            print('- query:', test_seq)
            print('- label:', test_labels)
            print('- predict:', prediction)
            print('====================')

        if save is not False:
            if not os.path.isdir(save):
                os.makedirs(save, exist_ok=True)
            prediction = prediction.cpu().numpy()[0]
            vector_to_binary(prediction, data_path=save, savefile=str(idx))


def train_transformer_model(model, train_ds, epochs, maxlen, early_stop_patience=None):
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()
            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, maxlen - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                             reduction='none')(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)

    def accuracy(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, maxlen - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    learning_rate = CustomSchedule(128)
    optimizer = tf.keras.optimizers.Adamax(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    if early_stop_patience is not None:
        early_stopping = EarlyStopping(monitor='loss', patience=early_stop_patience, verbose=True)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    if early_stop_patience is not None:
        model.fit(train_ds, epochs=epochs, callbacks=[early_stopping])
    else:
        model.fit(train_ds, epochs=epochs)

    return model


def test_transformer_model(model, test_ds, maxlen, verbose=False, save=False):
    def predict(model, sentence, maxlen):
        START_TOKEN, END_TOKEN = [SOS], [EOS]
        output = tf.expand_dims(START_TOKEN, 0)
        for i in range(maxlen):
            predictions = model(inputs=[sentence, output], training=False)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            if tf.equal(predicted_id, END_TOKEN[0]): break
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0)

    for idx, (test_seq, test_labels) in enumerate(test_ds):
        print('[*]', idx)
        prediction = predict(model, test_seq, maxlen)
        if verbose is True:
            print('====================')
            print('- query:', test_seq)
            print('- label:', test_labels)
            print('- predict:', prediction)
            print('====================')
        if save is not False:
            if not os.path.isdir(save):
                os.makedirs(save, exist_ok=True)
            prediction = prediction.cpu().numpy()
            vector_to_binary(prediction, data_path=save, savefile=str(idx))