from typing import Tuple, Dict, Any

import tensorflow as tf
from keras.api._v2 import keras
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPool2D
from keras.models import Model

import params


class ConvBlock:
    def __init__(
            self, 
            conv1_filters: int, 
            conv1_kernel_size: Tuple[int, int],
            conv2_filters: int, 
            conv2_kernel_size: Tuple[int, int], 
            maxpool_size: int, 
            dropout_rate: float) -> None:
        self.conv1: Conv2D = Conv2D(filters=conv1_filters, kernel_size=conv1_kernel_size, activation='relu', padding='same')
        self.conv2: Conv2D = Conv2D(filters=conv2_filters, kernel_size=conv2_kernel_size, activation='relu', padding='same')
        self.maxpool: MaxPool2D = MaxPool2D(pool_size=maxpool_size)
        self.dropout: Dropout = Dropout(rate=dropout_rate)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class CNNModel:
    def __init__(
            self,
            input_shape: Tuple[int, ...] = params.INPUT_SHAPE,
            conv_params1: Dict[str, Any] = params.CONV_PARAMS,
            conv_params2: Dict[str, Any] = params.CONV_PARAMS,
            dense_units: int = params.DENSE_UNITS,
            dropout_rate: float = params.DROPOUT_RATE,
            output_units: int = params.OUTPUT_UNITS) -> None:
        self.input: Input = Input(shape=input_shape)
        self.convblock1: ConvBlock = ConvBlock(**conv_params1)
        self.convblock2: ConvBlock = ConvBlock(**conv_params2)
        self.flatten: Flatten = Flatten()
        self.dense: Dense = Dense(units=dense_units, activation='relu')
        self.dropout: Dropout = Dropout(rate=dropout_rate)
        self.output: Dense = Dense(units=output_units, activation='softmax')

    def build(self) -> Model:
        input = self.input
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        output = self.output(x)
        return Model(inputs=input, outputs=output)
