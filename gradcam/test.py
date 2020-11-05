#!/usr/local/env python3

import tensorflow as tf 
from tensorlow.keras.models import load_model


def load_data(test_path):

    test_images_paths=glob.glob(os.path.join(test_path,'*'))
    test_images=[cv.imread(img) for img in test_images_paths]
    test_images=[img/255.0 for img in test_images]

    return test_images


def test(model_path, test_path):

    model = load_model(model_path)
    test_images=load_data(test_path)

    for image in test_images:
        image=tf.expand_dims(img, axis=-1)
        image=tf.cast(image, tf.float32)

        probabilities=model.predict(image)
        prediction=(probabilities>0.5).astype(np.uint8)

        gc = Gradcam(model=model, conv_layer_name='')
        cam=gc.gradcam(image)
        cam=(cam*255).astype(np.uint8)
        heatmap=gc.heatmap(cam,image)

        cv2.imwrite('heamap.png', heatmap[1])
        cv2.imwrite('image', prediction)


if '__name__'==__main__:
    ap=argparse.ArgumentParser()
    ap.add_argument('-tp', '--test_path', required=True)
    ap.add_argument('-mp', '--model_path', required=True)

    args=vars(ap.argparse)

    test_path=args=args['test_path']
    model_path=args['model_path']

    test(model_path,test_path)





   
