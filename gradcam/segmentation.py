#!/usr/local/env python3

'''
segmentation.py: segmentation of ct scans
'''

import tensorflow as tensorflow 
from tensorflow.keras.optimizers import Adam

from models import unet
from utilities.evaluation import diceCoef


def load_dataset(path):

   train_image_paths = glob.glob('annotations_prepped_train/*')
   train_mask_paths = glob.glob('images_prepped_train/*')
   test_mask_paths = glob.glob('annotations_prepped_test/*')
   test_image_paths = glob.glob('images_prepped_test/*')

   train_images = [[cv2.imread(img_file)] for img_file in train_image_paths]
   train_masks = [[cv2.imread(m_file)] for m_file in train_mask_paths]  
   test_images = [[cv2.imread(img_file)] for img_file in test_image_paths]
   test_masks = [[cv2.imread(m_file)] for m_file in test_mask_paths]  
   
   n_classes=max(list(np.unique(train_images)))
   x_train=np.vstack(images)
   y_train=np.vstack(masks)

   x_train=x_train/255.0
   x_test=x_test/255.0

   y_train=tf.one_hot(y_train,depth=n_classes)
   y_test=tf.one_hot(y_test, depth=n_classes)
   
   print('n_classes: {}'.format(n_classes))
   return x_train, x_test, y_train, y_test


def main(batch, epochs):

    x_train, x_test, y_train, y_test=load_dataset(path)
    
    model=UnetFunc(nOutput=n_classes,finaActivation='softmax')
    model.compile(optimizer, loss='categorical_crossentropy', metrics=[diceCoef])


    gen=ImageDataGenerator(
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          horizontal_flip=True,
                          brightness_range=[0.5,1.5]
                          )

    train_gen=gen.flow(x_train, y_train, batch_size=batch)
    valid_gen=gen.flow(x_test,y_test)

    train_steps=len(x_train)/batch
    valid_steps=len(x_test)/batch

    history=model.fit(train_gen,
                      steps_per_epoch=train_steps,
                      epochs=epochs,
                      validation_data=valid_gen,
                      validation_steps=valid_steps,
                      verbose=0
                     )

    model.save('model.h5')


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', default=100, required=True)
    ap.add_argument('-b', '--batch', default=64, required=True)

    args=vars(ap.parse_args())

    batch=int(args['batch'])
    epochs=int(args['epochs'])
    main(batch, epochs)




    




