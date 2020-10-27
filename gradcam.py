import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K

tf.config.experimental_run_functions_eagerly(True)


class GradCam():

    def __init__(self, _class=None,  pixels=None, fMaps=None, model=None, 
            layer=None, lastlayer=None):

        self.pixels=None
        self.fMaps=None
        self.model=model
        self.layer=layer
        self._lastlayer=lastlayer
        self._class=_class


    @property
    def lastlayer(self):
        return model.layers[-1].name


    def gradients(self, logits, fmap):
        with tf.GradientTape() as tape:
            gradients=tape.gradient(logits,fmap)
        return gradients


    def weights(self):

        logits=model.get_layer('conv2d_6').output[:,:,:,0]
        logits_ij=logits*self.pixels
        fmap=model.get_layer('conv2d_18').output 
        g = self.gradients(logits_ij, fmap)
        
        return g


    def weightedMap(self):
        pass





model=load_model('unet_germ_2.5x_adam_weightedBinaryCrossEntropy_FRC_17_40.h5')
gc = GradCam()

grad=gc.weights()

print(grad)





