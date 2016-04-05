import numpy as np
import cv2
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
from keras import backend as K
import time
from FaceRec.pretrained_cnn import *
from keras.utils import np_utils

PRETRAINED = "cnn_weights.h5"


def graphnet():
	graph = Graph()
	graph.add_input(name='input', input_shape=(3, 227, 227))
	graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
		name='conv1_1')
	graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
		name='conv1_2')
	graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)))

	graph.add_node(Convolution2D(128, 3, 3, activation='relu', border_mode='same'),
		name='conv2_1')
	graph.add_node(Convolution2D(128, 3, 3, activation='relu', border_mode='same'),
		name='conv2_2')
	graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)))

	graph.add_node(Convolution2D(256, 3, 3, activation='relu', border_mode='same'),
		name='conv3_1')
	graph.add_node(Convolution2D(256, 3, 3, activation='relu', border_mode='same'),
		name='conv3_2')
	graph.add_node(Convolution2D(256, 3, 3, activation='relu', border_mode='same'),
		name='conv3_3')
	graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)))

	graph.add_node(Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
		name='conv4_1')
	graph.add_node(Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
		name='conv4_2')
	graph.add_node(Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
		name='conv4_3')
	graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)))

	graph.add_node(Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
		name='conv5_1')
	graph.add_node(Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
		name='conv5_2')
	graph.add_node(Convolution2D(512, 3, 3, activation='relu', border_mode='same'),
		name='conv5_3')
	graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)))

	print 'Loading weights ...'
    start=time.time()
    f = h5py.File(PRETRAINED)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print 'Model loaded in ..',time.time()-start


	# graph.compile(optimizer='rmsprop', loss={'output1':'mse', 'output2':'mse'})
	# history = graph.fit({'input':X_train, 'output1':y_train, 'output2':y2_train}, nb_epoch=10)
