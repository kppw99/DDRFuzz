import os

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape


# Size of the noise vector
NOISE_DIM = 128


def get_discriminator(data_dim):
    input_data = Input(shape=(data_dim,))
    x = Dense(512, kernel_initializer='he_normal', activation='relu')(input_data)
    x = Dense(256, activation='relu')(x)
    x = Dense(1)(x)
    return Model(input_data, x, name='discriminator')


def get_generator(data_dim):
    noise = Input(shape=(NOISE_DIM,))
    x = Dense(256, kernel_initializer='he_normal', activation='relu')(noise)
    x = Dense(512, activation='relu')(x)
    x = Dense(data_dim, activation='tanh')(x)
    return Model(noise, x, name='generator')


def discriminator_loss(real_data, fake_data):
    real_loss = tf.reduce_mean(real_data)
    fake_loss = tf.reduce_mean(fake_data)
    return fake_loss - real_loss


def generator_loss(fake_data):
    return -tf.reduce_mean(fake_data)


class WGAN(Model):
    def __init__(self, discriminator, generator, latent_dim,
                 discriminator_extra_steps=3, gp_weight=10.0):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_data, fake_data):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_data - real_data
        interpolated = real_data + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_data):
        if isinstance(real_data, tuple):
            real_data = real_data[0]

        # Get the batch size
        batch_size = tf.shape(real_data)[0]

        # Train the discriminator first.
        # The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_data = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_data, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_data, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_data=real_logits, fake_data=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_data, fake_data)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size,
                                                        self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


if __name__=='__main__':
    from util import *
    # ===============================================================================
    # Set training parameter
    DATA_DIM = 1000

    EPOCHS = 100
    BATCH_SIZE = 32
    SEED_COUNT = 1000

    DATA_PATH   = '../seq2seq/init_dataset/MP3/sample/'
    MODEL_PATH  = './model/wgan/mp3_weight'
    SEED_PATH   = './output/MP3/wgan/'

    MODE = 'train'  # train or generate
    # ===============================================================================

    # ===============================================================================
    # Load WGAN Model and Generate data
    # random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
    # generated_images = self.model.generator(random_latent_vectors)
    # generated_images = (generated_images * 127.5) + 127.5

    if MODE == 'generate':
        d_model = get_discriminator(data_dim=DATA_DIM)
        g_model = get_generator(data_dim=DATA_DIM)
        loaded_wgan = WGAN(discriminator=d_model, generator=g_model, latent_dim=NOISE_DIM)
        loaded_wgan.load_weights(MODEL_PATH)

        for idx in range(SEED_COUNT):
            random_vectors = tf.random.normal(shape=(1, NOISE_DIM))
            generated_data = loaded_wgan.generator(random_vectors)
            generated_data = (generated_data * NORM_COEF) + NORM_COEF
            generated_data = generated_data.numpy()
            generated_data = generated_data.astype(int)[0]
            if not os.path.isdir(SEED_PATH):
                os.makedirs(SEED_PATH, exist_ok=True)
            vector_to_binary(generated_data, data_path=SEED_PATH, savefile=str(idx))
    # ===============================================================================

    # ===============================================================================
    # Build and Train WGAN Model
    else:
        train_data, _ = load_dataset(DATA_PATH, pad_maxlen=DATA_DIM)
        train_data = (train_data - NORM_COEF) / NORM_COEF  # data normaliaze [-1.0, 1.0]

        d_model = get_discriminator(data_dim=DATA_DIM)
        d_model.summary()

        g_model = get_generator(data_dim=DATA_DIM)
        g_model.summary()

        generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

        wgan = WGAN(discriminator=d_model, generator=g_model, latent_dim=NOISE_DIM)
        wgan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer,
                     g_loss_fn=generator_loss, d_loss_fn=discriminator_loss)

        wgan.fit(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS)
        wgan.save_weights(MODEL_PATH)
    # ===============================================================================
