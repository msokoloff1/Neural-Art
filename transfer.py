##Imports##
import net
import utilities as utils
import tensorflow as tf
from functools import reduce
import numpy as np
import time
import argparse
###########################




parser = argparse.ArgumentParser(description='Parameters for the neural art algorithm')
parser.add_argument('-save_path'         , default = './results'                                            , help = 'Path to directory containing images to be stylized')
parser.add_argument('-train_iters'       , default = 1500, type=int                                         , help = 'Number of iterations for training operation')
parser.add_argument('-style_image_path'  , default = 'images/style0.jpg'                                    , help = 'Image to copy style from')
parser.add_argument('-content_image_path', default = 'images/content1.jpg'                                  , help = 'Image to copy content from')
parser.add_argument('-style_layers'      , default = ['conv1_2','conv2_2','conv3_3', 'conv4_1', 'conv5_1']  , help = 'Which layers of the vgg network to be used for obtaining style statistics')
parser.add_argument('-style_weights'     , default = [0.2      ,0.2     , 0.3     , 0.3       , 0.2      ]  , help = 'Weights for the loss between generator result and style image for each layer in the vgg network')
parser.add_argument('-tvnorm_weight'     , default = 1.5 , type=float                                       , help = 'Weight for the tv norm loss')
parser.add_argument('-style_weight'      , default = 0.0001, type=float                                     , help = 'Weight for the style loss')
parser.add_argument('-content_weight'    , default = 0.05, type=float                                       , help = 'Weight for the content loss')           
parser.add_argument('-result_shape'      , default = (int(720),int(1280),3)                                 , help = 'Dimensions of stylized result (Height/Width/Color Channels)')
parser.add_argument('-learning_rate'     , default = 0.025                                                  , help = 'The learning rate to be used when applying gradients')
parser.add_argument('-style_name'        , default = 'style0'                                               , help = 'Name of the style source')
parser.add_argument('-content_name'      , default = 'content1'                                             , help = 'Name of the content source')
args = parser.parse_args()

nameAppend = args.style_name + args.content_name
##Global Options##
contentPath        = args.content_image_path
stylePath          = args.style_image_path
contentLayer       = 'conv4_2'
destDir            =  args.save_path
styleLayers        = args.style_layers
styleWeights       = args.style_weights
styleData          = {}
styleBalanceData   = {}
contentData        = None
errorMetricContent = utils.mse #utils.euclidean
errorMetricStyle   = utils.mse
normalizeContent   = True #Inversion paper says to use this
normalizeStyle     = False #No mention of style in inversion paper
imageShape         = args.result_shape 
#TODO : Add a real value for sigma. Should be the average euclidean norm of the vgg training images. This also requires an option for the model to change whether or not sigma is multipled by the input image
sigma = 1.0 #<- if sigma is one then it doesnt need to be included in the vgg net (because the multiplicative identity)
beta = 2.0
alpha = 6.0
a = 0.01
B = 120.0 #Pixel min/max encourages pixels to be in the range of [-B, +B]

alphaNormLossWeight = 0.0001
TVNormLossWeight    = args.tvnorm_weight
styleLossWeight     = args.style_weight 
contentLossWeight   = args.content_weight 

learningRate  = args.learning_rate
numIters      = args.train_iters
showEveryN    = 500
##################


with tf.Session() as sess:
    inputTensor = tf.placeholder(tf.float32, shape=[None,imageShape[0], imageShape[1], imageShape[2]])
    contentImage = np.array(utils.loadImage(contentPath, imageShape))
    styleImage = utils.loadImage(stylePath, imageShape)
    styleBalanceImage = utils.loadImage(contentPath, imageShape)
    model =net.Vgg19()
    model.build(inputTensor, imageShape)
    contentData = eval('sess.run(model.' + contentLayer + ',feed_dict={inputTensor:contentImage})')
    for styleLayer in styleLayers:
        styleData[styleLayer] = np.array(eval('sess.run(model.' + styleLayer + ',feed_dict={inputTensor:styleImage})'))





def buildStyleLoss(model):
    totalStyleLoss = []
    for index, styleLayer in enumerate(styleLayers):
        normalizingConstant = 1
        if (normalizeStyle):
            normalizingConstant = (reduce(lambda x, y: x + y, (styleData[styleLayer] ** 2)) ** (0.5))

        styleLayerVar = tf.Variable(styleData[styleLayer])
        correctGrams  = buildGramMatrix(styleLayerVar)
        tensorGrams   = buildGramMatrix(eval('model.'+styleLayer))
        _, dimX, dimY, num_filters = styleLayerVar.get_shape()
        denominator   =(2*normalizingConstant)*((float(int(dimX))*float(int(dimY)))**2)*(float(int(num_filters))**2)
        error         = tf.reduce_sum(errorMetricStyle(tensorGrams, correctGrams))
        totalStyleLoss.append((tf.div(error,denominator)))


    #styleLoss = (reduce(lambda x, y: x + y, totalStyleLoss))
    styleLoss = tf.reduce_sum(totalStyleLoss)
    return styleLoss


def buildGramMatrix(layer):
    _, dimX, dimY, num_filters = layer.get_shape()
    vectorized_maps = tf.reshape(layer, [int(dimX) * int(dimY), int(num_filters)])

    if int(dimX) * int(dimY) > int(num_filters):
        return tf.matmul(vectorized_maps, vectorized_maps, transpose_a=True)
    else:
        return tf.matmul(vectorized_maps, vectorized_maps, transpose_b=True)


def buildContentLoss(model, correctAnswer=contentData):

    normalizingConstant = 1
    if(normalizeContent):

        normalizingConstant = np.sum(  (correctAnswer**2))**(0.5)

    print("Normalizing Constant : %g"%(normalizingConstant))
    contentLoss = (eval('errorMetricContent(model.' + contentLayer + ', correctAnswer)') / normalizingConstant)
    return tf.reduce_sum(contentLoss)


def buildAlphaNorm(model):
    adjustedImage = model.bgr
    return tf.reduce_sum(tf.pow(adjustedImage, alpha))



def buildTVNorm(model):
    adjustedImage = model.bgr


    yPlusOne = tf.slice(adjustedImage, [0,0,1,0], [1,imageShape[0],(imageShape[1]-1),imageShape[2]])
    xPlusOne = tf.slice(adjustedImage, [0,1,0,0], [1,(imageShape[0]-1),imageShape[1],imageShape[2]])

    inputNoiseYadj = tf.slice(adjustedImage,[0,0,0,0],[1,imageShape[0],(imageShape[1]-1),imageShape[2]])
    inputNoiseXadj = tf.slice(adjustedImage, [0,0,0,0], [1,(imageShape[0]-1),imageShape[1],imageShape[2]])


    lambdaBeta = (sigma**beta) / (imageShape[0]*imageShape[1]*((a*B)**beta))
    error1 = tf.slice(tf.square(yPlusOne-inputNoiseYadj), [0,0,0,0], [1,(imageShape[0]-1),(imageShape[1]-1), imageShape[2]])
    error2 = tf.slice(tf.square(xPlusOne-inputNoiseXadj), [0,0,0,0], [1,(imageShape[0]-1),(imageShape[1]-1), imageShape[2]])

    return lambdaBeta*tf.reduce_sum( tf.pow((error1+error2),(beta/2) ))



def totalLoss(model):
    #errorComponents = [buildAlphaNorm(model), buildTVNorm(model), buildStyleLoss(model), buildContentLoss(model)]
    #LossWeights = [alphaNormLossWeight, TVNormLossWeight, styleLossWeight, contentLossWeight]
#PROBLEM buildTVNorm
    errorComponents =[buildStyleLoss(model), buildContentLoss(model), buildTVNorm(model)]
    LossWeights = [styleLossWeight, contentLossWeight,TVNormLossWeight]
    loss =[]
    for error, weights in zip(errorComponents, LossWeights):
        loss.append(error*weights)

    reducedLoss = reduce(lambda x,y: x+y, loss)
    return reducedLoss

def getUpdateTensor(model, inputVar):
    loss = totalLoss(model)
    #tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': 100}).minimize(session)    
    optimizer = tf.train.AdamOptimizer(learningRate)
    grads = optimizer.compute_gradients(loss, [inputVar])
    clipped_grads = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in grads]
    return [optimizer.apply_gradients(clipped_grads), loss]


def train(model, inputVar, sess):
    updateTensor, lossTensor = getUpdateTensor(model, inputVar)
    sess.run(tf.initialize_all_variables())
    start_time = time.time()

    for iteration in range(numIters):
        sess.run(updateTensor)
        if(iteration%showEveryN==0):
            img = inputVar.eval()
            print("Iteration : %s | Loss : %g"%(str(iteration).zfill(4), lossTensor.eval()))
            utils.showImage(img,imageShape, destDir, str(iteration)+nameAppend)
        elif(iteration%10==0):
            print("Iteration : %s | Loss : %g" % (str(iteration).zfill(4), lossTensor.eval()))

    elapsed = time.time() -start_time
    print("Experiment Took : %s"%(str(elapsed)))



with tf.Session() as sess:
    model = net.Vgg19()
    inputVar = tf.Variable(tf.random_uniform((1,)+imageShape, minval=0.25, maxval=0.75))
    model.build(inputVar, imageShape)
    train(model, inputVar, sess)
    img =inputVar.eval()
    utils.showImage(img,imageShape,  destDir, 'final'+nameAppend)
