import tensorflow as tf
import numpy as np
import transfer as t

def mse(preds, correct):
    tf.reduce_mean()

def euclidean(preds, correct):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(preds, correct))))


def createNoiseImage(shape=(1,224,224,3)):
    return np,random.random_sample(shape)

def loadImage(path, display=False):
    subjectImage = misc.imresize(misc.imread(path), (224, 224, 3)) / 255
    if(display):
        showImage(subjectImage)


def showImage():
    img = Image.fromarray((subjectImage * 255).astype('uint8'), 'RGB')
    img.show()
