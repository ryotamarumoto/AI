import cv2
import pathlib
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
import glob


def load_image(img_path):

    data = []

    for path in glob.glob(img_path):
        img = Image.open(path)
        img = np.array(img.convert("RGB").resize((256, 256)))
        data.append(img)
    return np.stack(data, axis=0)

def load_screw():
    x = load_image("/Users/ryotamarumoto/Work/corpy/archive/dataset/train/good/*.png")
    y = load_image("/Users/ryotamarumoto/Work/corpy/archive/dataset/test/*.png")

    # split normal images into train and test data
    X_train, X_test = train_test_split(x, test_size=0.2, random_state=0)
    return X_train, X_test, y