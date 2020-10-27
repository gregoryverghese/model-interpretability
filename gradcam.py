import cv2
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


    def weights(self, image):
        
        image=tf.expand_dims(image, axis=0)
        logits=model.get_layer('conv2d_6').output[:,:,:,self._class]
        #logits_ij=logits*self.pixels
        fmap=model.get_layer('conv2d_18').output 
        g = self.gradients(logits, fmap)

        gfunction=K.function([self.model.input],[logits,fmap])   
        x, grads=gfunction([image])

        
        return x,grads


    def weightedMap(self):
        self.alpha=tf.mean(self.grads)
        self.alpha=abs(self.alpha)
        return self.alpha

    
    def activationMaps():

        cam=tf.dot(self.A, self.alpha_c)
        return cam



    def sgd(self, image):

        weights=self.weights(image)
        return weights
        






image=cv2.imread('48.90239 C L1.3.png')
model=load_model('unet_germ_2.5x_adam_weightedBinaryCrossEntropy_FRC_17_40.h5')
gc = GradCam(model=model)

grad=gc.weights(image)

print(grad)





