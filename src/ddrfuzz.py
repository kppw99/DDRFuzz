#!/usr/bin/python3

import os
import argparse
from util import *
from models import *
from wgan import *
from seq2seq import *
from attention import *
from transformer import *
from keras.utils import plot_model


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str,
                        help='[seq2seq | attention | transformer | wgan]',
                        default='wgan')
    parser.add_argument('--mode', dest='mode', type=str,
                        help='[train | test]', default='test')
    parser.add_argument('--path', dest='path', type=str,
                        default='../seq2seq/init_dataset/MP3/sample/')
    parser.add_argument('--maxlen', dest='maxlen', type=int, default=1000)
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=64)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--patience', dest='patience', type=int, default=10)
    parser.add_argument('--stop_acc', dest='stop_acc', type=float, default=0.3)
    args = parser.parse_args()

    MAXLEN = args.maxlen
    EMBEDDING_DIM = args.emb_dim
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    PATIENCE = args.patience
    DATASET_PATH = args.path
    STOP_ACC = args.stop_acc

    print('\n[*] Setting Arguments')
    print('- MODEL:', args.model)
    print('- MODE:', args.mode)
    print('- MAXLEN:', MAXLEN)
    print('- BATCH_SIZE:', BATCH_SIZE)
    print('- EPOCHS:', EPOCHS)
    print('- PATIENCE:', PATIENCE)
    print('- EMBEDDING_DIM:', EMBEDDING_DIM)
    print('- DATASET_PATH:', DATASET_PATH)
    print('- STOP_ACC:', STOP_ACC)

    input_tensor, target_tensor = load_dataset(DATASET_PATH, pad_maxlen=MAXLEN)

    # Seq2Seq Model
    if args.model == 'seq2seq':
        print('\n[*] Start simple seq2seq model ...')

        train_ds, test_ds = split_tensor(input_tensor, target_tensor, batch_size=BATCH_SIZE, test_ratio=0.0)
        model = Seq2seq(ENC_VOCAB_SIZE, DEC_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                        units=256, sos=SOS, eos=EOS, maxlen=MAXLEN)

        dir_path = './model/s2s/'
        filename = 'simple_s2s_model_latest'
        fullname = os.path.join(dir_path, filename)

        if args.mode == 'train':
            model = train_seq2seq_model(model, train_ds, EPOCHS, early_stop_patience=PATIENCE, stop_acc=STOP_ACC)
            if (os.path.isdir(dir_path) == False):
                os.mkdir(dir_path)

            cpfile = os.path.join(dir_path, 'checkpoint')
            if (os.path.isfile(cpfile) == True):
                filetime = time.strftime("_%Y%m%d-%H%M%S")
                newcp = cpfile + filetime
                fullname1 = fullname + '.index'
                fullname2 = fullname + '.data-00000-of-00001'
                newname1 = fullname + filetime + '.index'
                newname2 = fullname + filetime + '.data-00000-of-00001'

                os.rename(cpfile, newcp)
                os.rename(fullname1, newname1)
                os.rename(fullname2, newname2)

            model.save_weights(fullname)
        else:
            model.load_weights(fullname)
            test_seq2seq_model(model, test_ds, verbose=False, save='./output/PNG/seq2seq/')
    # Attention Model
    elif args.model == 'attention':
        print('\n[*] Start seq2seq with attention model ...')

        train_ds, test_ds = split_tensor(input_tensor, target_tensor, batch_size=BATCH_SIZE, test_ratio=0.0)
        model = Attentions(ENC_VOCAB_SIZE, DEC_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, units=256,
                           sos=SOS, eos=EOS, maxlen=MAXLEN)

        dir_path = './model/attention'
        filename = 'attention_model_latest'
        fullname = os.path.join(dir_path, filename)

        if args.mode == 'train':
            model = train_seq2seq_model(model, train_ds, EPOCHS, early_stop_patience=PATIENCE, stop_acc=STOP_ACC)
            if (os.path.isdir(dir_path) == False):
                os.mkdir(dir_path)

            cpfile = os.path.join(dir_path, 'checkpoint')
            if (os.path.isfile(cpfile) == True):
                filetime = time.strftime("_%Y%m%d-%H%M%S")
                newcp = cpfile + filetime
                fullname1 = fullname + '.index'
                fullname2 = fullname + '.data-00000-of-00001'
                newname1 = fullname + filetime + '.index'
                newname2 = fullname + filetime + '.data-00000-of-00001'

                os.rename(cpfile, newcp)
                os.rename(fullname1, newname1)
                os.rename(fullname2, newname2)

            model.save_weights(fullname)
        else:
            model.load_weights(fullname)
            test_seq2seq_model(model, test_ds, verbose=False, save='./output/PNG/attention/')
    # Transformer Model
    elif args.model == 'transformer':
        print('\n[*] Start transformer model ...')

        train_ds, test_ds = split_tensor(input_tensor, target_tensor, batch_size=BATCH_SIZE,
                                         algo='transformer', test_ratio=0.0)

        model = transformer(input_dim=MAXLEN, num_layers=4, dff=512,
                            d_model=EMBEDDING_DIM, num_heads=4, dropout=0.3, name="transformer")

        dir_path = './model/transformer'
        filename = 'transformer_model_latest'
        fullname = os.path.join(dir_path, filename)

        if args.mode == 'graph':
            plot_model(model, show_shapes=True, to_file='transformer_model_simple.png')
            plot_model(model, expand_nested=True, show_shapes=True, to_file='transformer_model.png')
            exit(0)

        if args.mode == 'train':
            model = train_transformer_model(model, train_ds, EPOCHS, MAXLEN, early_stop_patience=PATIENCE, stop_acc=STOP_ACC)
            if (os.path.isdir(dir_path) == False):
                os.mkdir(dir_path)

            cpfile = os.path.join(dir_path, 'checkpoint')
            if (os.path.isfile(cpfile) == True):
                filetime = time.strftime("_%Y%m%d-%H%M%S")
                newcp = cpfile + filetime
                fullname1 = fullname + '.index'
                fullname2 = fullname + '.data-00000-of-00001'
                newname1 = fullname + filetime + '.index'
                newname2 = fullname + filetime + '.data-00000-of-00001'

                os.rename(cpfile, newcp)
                os.rename(fullname1, newname1)
                os.rename(fullname2, newname2)

            model.save_weights(fullname)
        else:
            model.load_weights(fullname)
            test_transformer_model(model, test_ds, MAXLEN, verbose=False, save='./output/PNG/transformer/')
            # Transformer Model
    elif args.model == 'wgan':
        print('\n[*] Start wgan model ...')

        DATA_DIM = MAXLEN

        SEED_COUNT = 1000
        SEED_PATH = './output/MP3/wgan/'

        dir_path = './model/wgan'
        filename = 'wgan_model_latest'
        fullname = os.path.join(dir_path, filename)

        d_model = get_discriminator(data_dim=DATA_DIM)
        g_model = get_generator(data_dim=DATA_DIM)
        wgan = WGAN(discriminator=d_model, generator=g_model, latent_dim=NOISE_DIM)

        if args.mode == 'train':
            train_data = (input_tensor - NORM_COEF) / NORM_COEF  # data normaliaze [-1.0, 1.0]

            d_model.summary()
            g_model.summary()

            generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
            discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
            wgan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer,
                         g_loss_fn=generator_loss, d_loss_fn=discriminator_loss)

            wgan.fit(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS)

            if (os.path.isdir(dir_path) == False):
                os.mkdir(dir_path)

            cpfile = os.path.join(dir_path, 'checkpoint')
            if (os.path.isfile(cpfile) == True):
                filetime = time.strftime("_%Y%m%d-%H%M%S")
                newcp = cpfile + filetime
                fullname1 = fullname + '.index'
                fullname2 = fullname + '.data-00000-of-00001'
                newname1 = fullname + filetime + '.index'
                newname2 = fullname + filetime + '.data-00000-of-00001'

                os.rename(cpfile, newcp)
                os.rename(fullname1, newname1)
                os.rename(fullname2, newname2)

            wgan.save_weights(fullname)
        else:
            wgan.load_weights(fullname)

            for idx in range(SEED_COUNT):
                random_vectors = tf.random.normal(shape=(1, NOISE_DIM))
                generated_data = wgan.generator(random_vectors)
                generated_data = (generated_data * NORM_COEF) + NORM_COEF
                generated_data = generated_data.numpy()
                generated_data = generated_data.astype(int)[0]
                if not os.path.isdir(SEED_PATH):
                    os.makedirs(SEED_PATH, exist_ok=True)
                vector_to_binary(generated_data, data_path=SEED_PATH, savefile=str(idx))
    else:
        print('Please enter the right model name')
        print('python3 ./ddrfuzz.py --model [seq2seq|attention|transformer]')
