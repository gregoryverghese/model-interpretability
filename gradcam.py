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
        self._class=0
        self._alpha=None
        self._A=None


    @property
    def lastlayer(self):
        return model.layers[-1].name


    def calculate_alpha(self, image):

        with tf.GradientTape() as tape:
            image=tf.expand_dims(image,axis=0)
            conv_layer=model.get_layer(self.conv_name)
            A_k=conv_layer.output

            final_conv_layer=self.model.get_layer(self.final_conv_name)
            y=final_conv_layer.output
        
            gModel=Model([self.model.inputs], [A_k,y])
            
            convOutput, pred=gModel(image)
            y_c=pred[...,-1]
            grads = tape.gradient(y_c,convOutput)

        self.alpha=np.mean(grads, axis=(0,1,2))
        self._A=convOutput
        

    def gradcam(self, image):
        
        self.calculate_alpha(image)
        cam=np.dot(self._A, self.alpha)
        #cam=cam*(cam>0)
        cam=np.maximum(cam,0)
        x,y,_=image.shape
        cam=cv2.resize(cam,(x,y))
    
        return cam




image=cv2.imread('100042_01_LR_101536_62016.png')
model=load_model('unet_germ_2.5x_adam_weightedBinaryCrossEntropy_FRC_17_40.h5')

layer=model.layers[-1].name
gc = GradCam(model=model, final_conv_name=layer, conv_name='conv2d_9')

#grad=gc.calculate_alpha(image)
cam=gc.gradcam(image)



#print(grad)





