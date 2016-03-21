import numpy as np
import cv2
# import tensorflow
import theano
# import theano.tensor as T
# from theano.tensor.signal import conv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import normalization
import h5py
from test import *
from FaceRec.pretrained_cnn import *
from keras import backend as K


def VGGNet(X_train, y_train):

    first_layer = ZeroPadding2D((1, 1), input_shape=(3, 227, 227))
    # Sequential Model
    model = Sequential()
    model.add(first_layer)
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(output_dim=4096, activation='relu', init="orthogonal"))
    model.add(Dense(output_dim=4096, init="uniform", activation='relu'))
    model.add(Dense(output_dim=10, init="uniform"))
    model.add(Activation("softmax"))

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # cnn = pretrained_cnn()

    # for k in cnn[cnn.keys()[0]]:
    #     for i in k:
    #         a = i[0][0][1][0]
    #         if 'conv' in a:
    #             weight1 = np.rollaxis(i[0][0][2][0][0],3,start=0)
    #             weight1 = np.rollaxis(weight1,3,start=1)
    #             weight2 = np.rollaxis(i[0][0][2][0][1],1,start=0)[0]
    #             weights = [weight1,weight2]
    #             layer_dict[a].set_weights(weights)
    #             print "Weights added to",a

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='mean_squared_error', optimizer=adam)
    model.fit(X_train, y_train, nb_epoch=3, batch_size=16, verbose=1)

    # print model
