import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
from tensorflow.keras import Input
import tensorflow.keras.backend as K

tf.config.experimental_run_functions_eagerly(True)


class GradCam():

    def __init__(self, _class=None,  pixels=None, fMaps=None, model=None, 
            conv_name=None, final_conv_name=None):
        

        self.pixels=None
        self.conv_name=conv_name
        self.final_conv_name=final_conv_name
        self.model=model
        self._class=_class


    @property
    def lastlayer(self):
        return model.layers[-1].name


    def weights(self, image):
        
        image=tf.expand_dims(image, axis=0)

        conv_layer=model.get_layer(self.conv_name)
        conv_model=Model(self.model.input,conv_layer.output)
       
        final_input=Input(shape=conv_layer.output.shape)
        final_conv_layer=self.model.get_layer(self.final_conv_name)(final_input)
        final_conv_model=Model(final_input, final_conv_layer.output)

        with tf.GradientTape() as tape:
            final_conv_output=final_conv_model(image)
            tape.watch(final_conv_output)
            preds=final_conv_model(final_conv_output)
            gradients=tape.gradient(preds,final_conv_output)
        
        return gradients


    def weights2(self, image):

        image=tf.expand_dims(image,axis=0)
        conv_layer=model.get_layer(self.conv_name)



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
print(model.inputs)
gc = GradCam(model=model, final_conv_name='conv2d_18', conv_name='conv2d_9')

grad=gc.weights(image)

print(grad)





