#!/usr/local/env python3

"""
classification.py: simple cifar10 classification
task to test gradcam script
"""

import argparse

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from vgg_model import vgg_4block


def load_dataset():

    (trainX, trainY), (testX, testY) = cifar10.load_data()

    #scale values between 0-1
    trainX, testX = trainX/255.0, testX/255.0
    
    #one-hot encode labels
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, testX, trainY, testY 


    
def plot_history(history):

    f, ax = plt.subplots(1,2, figsize=(10,10))
    ax[0].plot(history.history['loss'], color='blue', label='train')
    ax[0].plot(history.history['loss'], color='orange', label='test')

    ax[1].plot(history.history['accuracy'], color='blue', label='train')
    ax[1].plot(history.history['accuracy'], color='orange', label='test')

    plt.savefig('learning curves.png')

 

def main(batch,epochs):

    trainX, testX, trainY, testY = load_dataset()
    shape = trainX.shape
    print('Dimensions: {}'.format(shape))
    model = vgg_4block(shape[1], shape[2])

    #Note we already perform scaling
    gen = ImageDataGenerator(
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            brightness_range=[0.5,1.5]
                            )

    train_gen = gen.flow(trainX, trainY, batch_size=64)
    valid_gen = gen.flow(testX, testY)

    print(len(trainX))

    train_steps = len(trainX)/batch 
    valid_steps = len(testX)/batch 

    print(model)
    history=model.fit_generator(train_gen, 
              steps_per_epoch=train_steps, 
              epochs=epochs,
              validation_data=valid_gen, 
              validation_steps=valid_steps,
              verbose=0
              )

    plot_history(history)



if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', default=100, required=True)
    ap.add_argument('-b', '--batch', default=64, required=True)
    
    args = vars(ap.parse_args())
    
    batch=int(args['batch'])
    epochs=int(args['epochs'])
    main(batch, epochs)



