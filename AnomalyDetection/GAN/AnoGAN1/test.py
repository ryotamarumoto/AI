import os, cv2
from dcgan import DCGAN
from anogan import ANOGAN
import numpy as np
from keras.optimizers import Adam
from loaddata import load_screw
from train import normalize, denormalize
from PIL import Image

if __name__ == '__main__':
    iterations = 100
    input_dim = 30
    anogan_optim = Adam(lr=0.001, amsgrad=True)

    ### 0. prepare data
    X_train, X_test, y = load_screw()
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    input_shape = X_train[0].shape

    ### 1. train generator & discriminator
    dcgan = DCGAN(input_dim, input_shape)
    dcgan.load_weights('/Users/ryotamarumoto/Work/corpy/models/GAN/AnoGAN2/weights/generator_9.h5', '/Users/ryotamarumoto/Work/corpy/models/GAN/AnoGAN2/weights/discriminator_9.h5')

    for i, test_img in enumerate(X_test):
        test_img = test_img[np.newaxis,:,:,:]
        anogan = ANOGAN(input_dim, dcgan.g)
        anogan.compile(anogan_optim)
        anomaly_score, generated_img = anogan.compute_anomaly_score(test_img, iterations)
        generated_img = denormalize(generated_img)
        imgs = np.concatenate((denormalize(test_img[0]), generated_img[0]), axis=1)
        # cv2.imwrite('/Users/ryotamarumoto/Work/corpy/models/GAN/AnoGAN2/predict' + os.sep + str(int(anomaly_score)) + '_' + str(i) + '.png', imgs)

        imgs = Image.fromarray(imgs)
        imgs.save('/Users/ryotamarumoto/Work/corpy/models/GAN/AnoGAN2/predict' + os.sep + str(int(anomaly_score)) + '_' + str(i) + '.png')

        print(str(i) + ' %.2f'%anomaly_score)
        with open('scores.txt', 'a') as f:
            f.write(str(anomaly_score) + '\n')