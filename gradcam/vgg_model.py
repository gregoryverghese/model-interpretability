#!/usr/local/env/ python3

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.layers import BatchNormaliztion, Flatten
from keras.layers import Dropout, Input

from keras.optimizer import Adam


conv2d_params={'kernel_size':(3,3), 
               'activation':'relu',
               'kernel_initializer':'he_uniform', 
               'padding':'same'}


def vgg_4block(x:

    x = Input((None,None,3))

    x = Conv2D(32, **conv2d_params)(x)
    x = BatchNorxmalization()(x)
    x = Conv2D(32, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, **conv2d_params)(x)
    x = BatchNormalization(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, **conv2d_params)(x)
    x = BatchNormaliztion(x)
    x = Conv2D(64, **conv2d_params)(x)
    x = BatchNormaliztion(x)
    x = Conv2D(64, **conv2d_params)(x)
    x = BatchNormaliztion(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, **conv2d_params)(x)
    x = BatchNormaliztion()(x)
    x = Conv2D(128, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, **conv2d_params)(x)
    x = BatchNormaliztion()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = Dense(10)(x)

    optimizer = Adam()
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model 
