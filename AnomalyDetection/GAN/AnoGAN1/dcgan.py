import math, cv2
import numpy as np
from generator import Generator
from discriminator import Discriminator
from keras.models import Model, Sequential
from keras.utils.generic_utils import Progbar


class DCGAN(object):
    def __init__(self, input_dim, image_shape):
        self.input_dim = input_dim
        self.d = Discriminator(image_shape).get_model()
        self.g = Generator(input_dim, image_shape).get_model()

    def compile(self, g_optim, d_optim):
        self.d.trainable = False
        self.dcgan = Sequential([self.g, self.d])
        self.dcgan.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.d.trainable = True
        self.d.compile(loss='binary_crossentropy', optimizer=d_optim)

    def train(self, epochs, batch_size, X_train):
        g_losses = []
        d_losses = []
        for epoch in range(epochs):
            np.random.shuffle(X_train)
            n_iter = X_train.shape[0] // batch_size
            progress_bar = Progbar(target=n_iter)
            for index in range(n_iter):
                # create random noise -> N latent vectors
                noise = np.random.uniform(-1, 1, size=(batch_size, self.input_dim))

                # load real data & generate fake data
                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                for i in range(batch_size):
                    if np.random.random() > 0.5:
                        image_batch[i] = np.fliplr(image_batch[i])
                    if np.random.random() > 0.5:
                        image_batch[i] = np.flipud(image_batch[i])
                generated_images = self.g.predict(noise, verbose=0)

                # attach label for training discriminator
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * batch_size + [0] * batch_size)

                # training discriminator
                d_loss = self.d.train_on_batch(X, y)

                # training generator
                g_loss = self.dcgan.train_on_batch(noise, np.array([1] * batch_size))

                progress_bar.update(index, values=[('g', g_loss), ('d', d_loss)])
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            if (epoch + 1) % 10 == 0:
                image = self.combine_images(generated_images)
                image = (image + 1) / 2.0 * 255.0
                cv2.imwrite('./result/' + str(epoch) + ".png", image)
            print('\nEpoch' + str(epoch) + " end")

            # save weights for each epoch
            # if (epoch + 1) % 50 == 0:
            self.g.save_weights('/Users/ryotamarumoto/Work/corpy/models/GAN/AnoGAN2/weights/generator_' + str(epoch) + '.h5', True)
            self.d.save_weights('/Users/ryotamarumoto/Work/corpy/models/GAN/AnoGAN2/weights/discriminator_' + str(epoch) + '.h5', True)
        return g_losses, d_losses

    def load_weights(self, g_weight, d_weight):
        self.g.load_weights(g_weight)
        self.d.load_weights(d_weight)

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
        shape = generated_images.shape[1:4]
        image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index / width)
            j = index % width
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img[:, :, :]
        return image
