# #!/usr/bin/python3
#
# import os
# import base64
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
#
#
# BASE64_DICT = {
#     'A': 64, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
#     'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20,
#     'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30,
#     'f': 31, 'g': 32, 'h': 33, 'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40,
#     'p': 41, 'q': 42, 'r': 43, 's': 44, 't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50,
#     'z': 51, '0': 52, '1': 53, '2': 54, '3': 55, '4': 56, '5': 57, '6': 58, '7': 59, '8': 60,
#     '9': 61, '+': 62, '/': 63,  # BASE64 (64)
#     '=': 0,  # Padding
#     '#': 65, # SOS
#     '!': 66  # EOS
# }
#
# ENC_VOCAB_SIZE = len(BASE64_DICT) - 2
# DEC_VOCAB_SIZE = len(BASE64_DICT)
# OUT_DIM = len(BASE64_DICT)
#
# SOS = BASE64_DICT['#']
# EOS = BASE64_DICT['!']
# PAD = BASE64_DICT['=']
#
#
# def binary_to_vector(filename, tag=0):
#     with open(filename, 'rb') as f:
#         binary_data = base64.b64encode(f.read()).decode('ascii')
#
#     if tag == 1:
#         binary_data = '#' + binary_data + '!'
#
#     return [BASE64_DICT[x] for x in binary_data]
#
#
# def vector_to_binary(vector_data, data_path='./dataset/', savefile=None):
#     reverse_base64_dict = dict(map(reversed, BASE64_DICT.items()))
#     ret_data = [reverse_base64_dict[x] for x in vector_data]
#     ret_data = ''.join(ret_data)
#     ret_data = base64.b64decode(ret_data)
#
#     if savefile is not None:
#         filename = os.path.join(data_path, savefile)
#         with open(filename, 'wb') as f:
#             f.write(ret_data)
#
#     return ret_data
#
#
# def get_input_files(dir_path):
#     files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
#              if os.path.isfile(os.path.join(dir_path, f))]
#     return [x for x in sorted(files) if '_prev' in x]
#
#
# def get_output_files(dir_path):
#     files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
#              if os.path.isfile(os.path.join(dir_path, f))]
#     return [x for x in sorted(files) if '_prev' not in x]
#
#
# def zero_padding(input_data, maxlen=None, tag=0):
#     data_list = list()
#     for data in input_data:
#         if len(data) < maxlen:
#             data = np.pad(data, (0, maxlen - len(data)))
#             if tag == 1:
#                 idx = list(data).index(0)
#                 data[idx] = BASE64_DICT['!']
#         else:
#             data = np.array(data[:maxlen])
#             if tag == 1:
#                 data[-1] = BASE64_DICT['!']
#         data_list.append(data)
#
#     return np.array(data_list)
#
#
# def load_dataset(path, pad_maxlen=None):
#     inputs = get_input_files(path)
#     outputs = get_output_files(path)
#
#     input_seeds, output_seeds = list(), list()
#     for in_file, out_file in zip(inputs, outputs):
#         input_seeds.append(binary_to_vector(in_file))
#         output_seeds.append(binary_to_vector(out_file, tag=1))
#
#     input_seeds = zero_padding(input_seeds, maxlen=pad_maxlen)
#     output_seeds = zero_padding(output_seeds, maxlen=pad_maxlen, tag=1)
#
#     return input_seeds, output_seeds
#
#
# def split_tensor(input_tensor, target_tensor, batch_size=64, test_ratio=0.2, algo=None):
#     if test_ratio != 0.0:
#         tr_X, te_X, tr_y, te_y = train_test_split(input_tensor, target_tensor, test_size=test_ratio)
#     else:
#         tr_X = input_tensor
#         tr_y = target_tensor
#
#     buffer_size = len(tr_X)
#
#     if algo is None:
#         train_ds = tf.data.Dataset.from_tensor_slices((tr_X, tr_y)).shuffle(buffer_size).batch(batch_size).prefetch(1024)
#         if test_ratio != 0.0:
#             test_ds = tf.data.Dataset.from_tensor_slices((te_X, te_y)).batch(1).prefetch(1024)
#         else:
#             test_ds = tf.data.Dataset.from_tensor_slices((tr_X, tr_y)).batch(1).prefetch(1024)
#     elif algo == 'transformer':
#         train_ds = tf.data.Dataset.from_tensor_slices((
#             {
#                 'inputs': tr_X,
#                 'dec_inputs': tr_y[:, :-1]
#             },
#             {
#                 'outputs': tr_y[:, 1:]
#             },
#         ))
#         train_ds = train_ds.cache()
#         train_ds = train_ds.shuffle(buffer_size)
#         train_ds = train_ds.batch(batch_size)
#         train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
#
#         if test_ratio != 0.0:
#             test_ds = tf.data.Dataset.from_tensor_slices((
#                 {
#                     'inputs': te_X,
#                     'dec_inputs': te_y[:, :-1]
#                 },
#                 {
#                     'outputs': te_y[:, 1:]
#                 },
#             ))
#         else:
#             test_ds = tf.data.Dataset.from_tensor_slices((
#                 {
#                     'inputs': tr_X,
#                     'dec_inputs': tr_y[:, :-1]
#                 },
#                 {
#                     'outputs': tr_y[:, 1:]
#                 },
#             ))
#         test_ds = test_ds.cache()
#         test_ds = test_ds.batch(1)
#         test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
#     else:
#         print('Error: please enter the right algorithm name!!!')
#         return None, None
#
#     return train_ds, test_ds
#
#
# if __name__=='__main__':
#     MAXLEN = 500
#     BATCH_SIZE = 64
#
#     input_tensor, target_tensor = load_dataset('../seq2seq/init_dataset/PNG/path', pad_maxlen=MAXLEN)
#     train_ds, target_ds = split_tensor(input_tensor, target_tensor, batch_size=BATCH_SIZE, test_ratio=0.0)
#
#     max_input_size, max_target_size = input_tensor.shape[1], target_tensor.shape[1]
#
#     print('[*] input_tensor.shape, target_tensor.shape:', input_tensor.shape, target_tensor.shape)
#     print('[*] max_input_size, max_target_size:', max_input_size, max_target_size)
#     print('[*] vocab_input_size, vocab_target_size:', ENC_VOCAB_SIZE, DEC_VOCAB_SIZE)

import os
import base64
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE64_DICT = {
    'A': 64, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
    'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20,
    'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30,
    'f': 31, 'g': 32, 'h': 33, 'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40,
    'p': 41, 'q': 42, 'r': 43, 's': 44, 't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50,
    'z': 51, '0': 52, '1': 53, '2': 54, '3': 55, '4': 56, '5': 57, '6': 58, '7': 59, '8': 60,
    '9': 61, '+': 62, '/': 63,  # BASE64 (64)
    '=': 0,  # Padding
    '#': 65,  # SOS
    '!': 66  # EOS
}

ENC_VOCAB_SIZE = len(BASE64_DICT) - 2
DEC_VOCAB_SIZE = len(BASE64_DICT)
OUT_DIM = len(BASE64_DICT)

SOS = BASE64_DICT['#']
EOS = BASE64_DICT['!']
PAD = BASE64_DICT['=']


def binary_to_vector(filename, tag=0):
    with open(filename, 'rb') as f:
        binary_data = base64.b64encode(f.read()).decode('ascii')

    if tag == 1:
        binary_data = '#' + binary_data + '!'

    return [BASE64_DICT[x] for x in binary_data]


def vector_to_binary(vector_data, data_path='./dataset/', savefile=None):
    reverse_base64_dict = dict(map(reversed, BASE64_DICT.items()))
    ret_data = [reverse_base64_dict[x] for x in vector_data]
    ret_data = ''.join(ret_data)
    ret_data = base64.b64decode(ret_data)

    if savefile is not None:
        filename = data_path + savefile
        with open(filename, 'wb') as f:
            f.write(ret_data)

    return ret_data


def get_input_files(dir_path):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f))]
    return [x for x in sorted(files) if '_prev' in x]


def get_output_files(dir_path):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f))]
    return [x for x in sorted(files) if '_prev' not in x]


def zero_padding(input_data, maxlen=None, tag=0):
    data_list = list()
    for data in input_data:
        if len(data) < maxlen:
            data = np.pad(data, (0, maxlen - len(data)))
            if tag == 1:
                idx = list(data).index(0)
                data[idx] = BASE64_DICT['!']
        else:
            data = np.array(data[:maxlen])
            if tag == 1:
                data[-1] = BASE64_DICT['!']
        data_list.append(data)

    return np.array(data_list)


def load_dataset(path, pad_maxlen=None):
    inputs = get_input_files(path)
    outputs = get_output_files(path)

    input_seeds, output_seeds = list(), list()
    for in_file, out_file in zip(inputs, outputs):
        input_seeds.append(binary_to_vector(in_file))
        output_seeds.append(binary_to_vector(out_file, tag=1))

    input_seeds = zero_padding(input_seeds, maxlen=pad_maxlen)
    output_seeds = zero_padding(output_seeds, maxlen=pad_maxlen, tag=1)

    return input_seeds, output_seeds


def split_tensor(input_tensor, target_tensor, batch_size=64, test_ratio=0.2, algo=None):
    if test_ratio != 0.0:
        tr_X, te_X, tr_y, te_y = train_test_split(input_tensor, target_tensor, test_size=test_ratio)
    else:
        tr_X = input_tensor
        tr_y = target_tensor

    buffer_size = len(tr_X)

    if algo is None:
        train_ds = tf.data.Dataset.from_tensor_slices((tr_X, tr_y)).shuffle(buffer_size).batch(batch_size).prefetch(
            1024)
        if test_ratio != 0.0:
            test_ds = tf.data.Dataset.from_tensor_slices((te_X, te_y)).batch(1).prefetch(1024)
        else:
            test_ds = tf.data.Dataset.from_tensor_slices((tr_X, tr_y)).batch(1).prefetch(1024)
    elif algo == 'transformer':
        train_ds = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': tr_X,
                'dec_inputs': tr_y[:, :-1]
            },
            {
                'outputs': tr_y[:, 1:]
            },
        ))
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(buffer_size)
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        if test_ratio != 0.0:
            test_ds = tf.data.Dataset.from_tensor_slices((
                {
                    'inputs': te_X,
                    'dec_inputs': te_y[:, :-1]
                },
                {
                    'outputs': te_y[:, 1:]
                },
            ))
        else:
            test_ds = tf.data.Dataset.from_tensor_slices((
                {
                    'inputs': tr_X,
                    'dec_inputs': tr_y[:, :-1]
                },
                {
                    'outputs': tr_y[:, 1:]
                },
            ))
        test_ds = test_ds.cache()
        test_ds = test_ds.batch(1)
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        print('Error: please enter the right algorithm name!!!')
        return None, None

    return train_ds, test_ds


if __name__ == '__main__':
    MAXLEN = 500
    BATCH_SIZE = 64

    input_tensor, target_tensor = load_dataset('../seq2seq/init_dataset/PNG/path', pad_maxlen=MAXLEN)
    train_ds, target_ds = split_tensor(input_tensor, target_tensor, batch_size=BATCH_SIZE, test_ratio=0.0)

    max_input_size, max_target_size = input_tensor.shape[1], target_tensor.shape[1]

    print('[*] input_tensor.shape, target_tensor.shape:', input_tensor.shape, target_tensor.shape)
    print('[*] max_input_size, max_target_size:', max_input_size, max_target_size)
    print('[*] vocab_input_size, vocab_target_size:', ENC_VOCAB_SIZE, DEC_VOCAB_SIZE)