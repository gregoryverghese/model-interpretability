import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
from tensorflow.keras import Input
import tensorflow.keras.backend as K


class GradCam():

    def __init__(self, model, convlayer_name, _class=0,  pixels=1):
        
        self.model=model
        self.pixels=pixels
        self._class=_class
        self.convlayer_name=convlayer_name
        self._alphas=None
        self._A=None
        self._y_c=None


    @property
    def finallayer(self):        
        return self.model.layers[-1]
        

    @property
    def convlayer(self):
        return self.model.get_layer(self.convlayer_name)


    @property
    def y_c(self):
        return self._y_c


    @y_c.setter
    def y_c(self, value):
        self._y_c=value
      

    @property
    def A(self):
        return self._A


    @A.setter
    def A(self, value):
        self._A=value


    def gradients(self, image):
        
        image=tf.expand_dims(image,axis=0)

        inputs=[self.model.inputs]
        outputs=[self.convlayer.output,self.finallayer.output]

        grad_model=Model(inputs, outputs)

        with tf.GradientTape() as tape:
            self.A, prediction = grad_model(image)
            print(prediction.shape)
            self.y_c = prediction[...,self._class]*pixels
            

            grads = tape.gradient(self.y_c, self.A)

        return grads


    def calculate_alphas(self, image):
        
        grads = self.gradients(image)
        self.alphas = tf.reduce_mean(grads, axis=(0,1,2))
        self.alphas = tf.maximum(self.alphas,0)

        return self.alphas

   
    def gradcam(self, image):
        
        self.calculate_alphas(image)
        cam=tf.multiply(self.A[0,:,:,:],self.alphas)
        cam=tf.reduce_sum(cam, axis=-1)

        cam=np.maximum(cam,0)/np.max(cam)   
        x,y,_ = image.shape
        cam=cv2.resize(cam,(x,y))

        return cam


    def heatmap(self, cam, image, colormap=cv2.COLORMAP_VIRIDIS, 
                alpha=0.5):

        heatmap=cv2.applyColorMap(cam,colormap)
        output=cv2.addWeighted(image, alpha, heatmap,1-alpha,0)
        return heatmap,output



'''
image=cv2.imread('14.90610 C L2.11.png')
model=load_model('unet_germ_2.5x_adam_weightedBinaryCrossEntropy_FRC_data_256_32_16_40.h5')

gc = GradCam(model=model, convlayer_name='conv2d_66')
cam=gc.gradcam(image)
cam=(cam*255).astype(np.uint8)
test=gc.heatmap(cam, image)
cv2.imwrite('test.png', test[0])
cv2.imwrite('test1.png', test[1])


for l in model.layers:
    print(l.name)
'''
