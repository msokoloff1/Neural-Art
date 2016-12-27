import tensorflow as tf
import numpy as np
from scipy import misc
from PIL import Image



def mse(preds, correct):
    return (tf.reduce_sum(tf.pow(tf.sub(preds,correct), 2)) / (2))

def euclidean(preds, correct):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.sub(preds, correct))))*(0.5)


def createNoiseImage(shape=(1,224,224,3)):
    return np.random.random_sample(shape).astype('float32')

def loadImage(path, display=False):
    subjectImage = misc.imresize(misc.imread(path), (224,224,3)) / 255
    if(display):
        showImage(subjectImage)

    return subjectImage.reshape(1,224,224,3)


def showImage(image):
    if(np.max(image)<1):
        image = image*255
    img = Image.fromarray((image).astype('uint8'), 'RGB')
    img.show()
