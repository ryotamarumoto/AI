from typing import Tuple, Optional, Union

import numpy as np
from nptyping import NDArray, Shape, Float, Int
import tensorflow as tf
from keras.api._v2 import keras
from keras.utils import to_categorical


# 型情報
Images = NDArray[Shape['Sample, Width, Height, Channel'], Int]
Labels = NDArray[Shape['Sample, 1'], Int]
PLabels = NDArray[Shape['Sample, Class'], Int]

def preprocess_dataset(
        images: Images, 
        labels: Optional[Labels] = None) -> Union[Images, Tuple[Images, PLabels]]:
    images: Images = images.astype('float32') / 255.0
    if labels is None:
        return images
    labels: Labels = to_categorical(labels, num_classes=10)
    return images, labels
