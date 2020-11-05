#!/usr/local/env/ python3

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Dropout, Input

from tensorflow.keras.optimizers import Adam


conv2d_params={'kernel_size':(3,3), 
               'activation':'relu',
               'kernel_initializer':'he_uniform', 
               'padding':'same'}


def vgg_4block(x_dim, y_dim):

    tensorinput = Input((x_dim, y_dim, 3))

    x = Conv2D(32, **conv2d_params)(tensorinput)
    x = BatchNormalization()(x)
    x = Conv2D(32, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, **conv2d_params)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, **conv2d_params)(x)
    x = BatchNormalization()(x)
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
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dense(10, activation='softmax')(x)

    optimizer = Adam()
    model = Model(tensorinput, x)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model 
