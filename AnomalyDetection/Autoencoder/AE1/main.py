from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import cv2
import glob

from tensorflow.python.ops.gen_nn_ops import LeakyRelu

# load data
train_img = glob.glob('/Users/ryotamarumoto/Work/corpy/archive/dataset/train/good/*')
train = []
for i in train_img:
    image= cv2.imread(i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    train.append(image)

# maybe if you need some data augmentation, do it here

train = np.array(train)
train = train.astype('float32') / 255

# hyper parameters
LEARNING_RATE = 0.0005
BATCH_SIZE = 8
Z_DIM = 100
EPOCHS = 50

# encoder

encoder_in = Input(shape=(300, 300, 3), name='encoder_input')
x = encoder_in
x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='encoder_conv_0')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='encoder_conv_0_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='encoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='encoder_conv_2')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='encoder_conv_3')(x)
x = LeakyReLU()(x)
shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)
encoder_out = Dense(Z_DIM, name='encoder_output')(x)
encoder = Model(encoder_in, encoder_out)

decoder_in = Input(shape=(Z_DIM,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_in)
x = Reshape(shape_before_flattening)(x)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_0')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_1')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', name='decoder_conv_t_2')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_2_5')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', name='decoder_conv_t_3')(x)
x = Activation('sigmoid')(x)

decoder_out = x
decoder = Model(decoder_in, decoder_out)

# connect in and out
model_in = encoder_in
model_out = decoder(encoder_out)
model = Model(model_in, model_out)

optimizer = Adam(learning_rate=LEARNING_RATE)


def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])


model.compile(optimizer=optimizer, loss=r_loss)

model.fit(train, train, batch_size=BATCH_SIZE, epochs=EPOCHS)