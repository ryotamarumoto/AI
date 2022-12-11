from keras.models import Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from keras.layers.core import Activation


class Generator(object):
    def __init__(self, input_dim, image_shape):
        INITIAL_CHANNELS = 3
        INITIAL_SIZE = 256

        inputs = Input((input_dim,))
        fc1 = Dense(input_dim=input_dim, units=INITIAL_CHANNELS * INITIAL_SIZE * INITIAL_SIZE)(inputs)
        fc1 = BatchNormalization()(fc1)
        fc1 = LeakyReLU(0.2)(fc1)
        fc2 = Reshape((INITIAL_SIZE, INITIAL_SIZE, INITIAL_CHANNELS), input_shape=(INITIAL_CHANNELS * INITIAL_SIZE * INITIAL_SIZE,))(fc1)
        up1 = Conv2DTranspose(64, (2, 2), strides=(1, 1), padding='same')(fc2)
        conv1 = Conv2D(64, (3, 3), padding='same')(up1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        up2 = Conv2DTranspose(64, (2, 2), strides=(1, 1), padding='same')(conv1)
        conv2 = Conv2D(image_shape[2], (5, 5), padding='same')(up2)
        outputs = Activation('tanh')(conv2)

        self.model = Model(inputs=[inputs], outputs=[outputs])

    def get_model(self):
        return self.model