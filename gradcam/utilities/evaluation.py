#/!usr/local/env python3

import tensorflow as tf 
import tensorflow.keras.backend as K



def dice_coef(y_pred, y_true, ax=[1,2,3],smooth=1):

    y_pred=tf.cast(y_pred, tf.float32)
    y_true=tf.cast(y_true, tf.float32)

    intersection=K.sum(y_pred*y_true,axis=ax)
    union=K.sum(y_pred, axis=ax) + K.sum(y_true, axis=ax)

    dice = K.mean((2*intersection + smooth)/(union+smooth), axis=0)

    return dice 



