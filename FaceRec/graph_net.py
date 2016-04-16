import numpy as np
import cv2
import keras
from keras.models import Sequential, Graph
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import normalization
import h5py
from test import *
from keras import backend as K
import time
from FaceRec.pretrained_cnn import *
from keras.utils import np_utils


def graphnet():
	
	model = Sequential()
    model.add(Convolution2D(64, 3, 3, input_shape=(3, 227, 227),
                            activation='relu', name='conv1_1', border_mode='same'))
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

    # rpn_graph = Graph()
    # rpn_graph.add_input(name='RPNinput', input_shape=(3, 227, 227))
    # for i in xrange(9):
    #     graph.add_node(Convolution2D(512, 3, 3, activation='relu'), name='rpn'+i, input='input1')    


	graph = Graph()
	graph.add_input(name='input1', input_shape=(3, 227, 227))
	graph.add_node(model, name='cnn', input='input1')
    #RPN Model to be added
    graph.add_node(Flatten(), input='cnn', name='flatten1')
    graph.add_node(Dense(output_dim=4096, activation='relu'), name='dense1', input='flatten1')
    graph.add_node(Dense(output_dim=4096, activation='relu'), name='dense2', input='dense1')
    graph.add_node(Dropout(0.5), name='dropout1', input='dense2')
    graph.add_node(Dense(output_dim=21, activation='softmax'), name='denseOut', input='dropout1')
    graph.add_output(name='output', input='denseOut')

	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv1_1', input='input1')
	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv1_2', input='conv1_1')
	# graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)), name='pool1', input='conv1_2')

	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv2_1', input='pool1')
	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv2_2', input='conv2_1')
	# graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)), name='pool2', input='conv2_2')

	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv3_1', input='pool2')
	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv3_2', input='conv3_1')
	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv3_3', input='conv3_2')
	# graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)), name='pool3', input='conv3_3')

	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv4_1', input='pool3')
	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv4_2', input='conv4_1')
	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv4_3', input='conv4_2')
	# graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)), name='pool4', input='conv4_3')

	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv5_1', input='pool4')
	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv5_2', input='conv5_1')
	# graph.add_node(Convolution2D(64, 3, 3, activation='relu', border_mode='same'), name='conv5_3', input='conv5_2')
	# graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)), name='pool5', input='conv5_3')

	layer_dict = dict([(layer.name, layer) for layer in graph.nodes['cnn'].layers])
	# print layer_dict

	PRETRAINED = "cnn_weights.h5"
    print 'Loading weights ...'
    start=time.time()
    f = h5py.File(PRETRAINED)
    for k in range(f.attrs['nb_layers']):
        if k >= len(graph.nodes['cnn'].layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        graph.nodes['cnn'].layers[k].set_weights(weights)
    f.close()
    print 'Model loaded in ..',time.time()-start

    print 'Normalizing data...'
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train = X_train - np.average(X_train)
    X_test = X_test - np.average(X_test)
    y_train = np_utils.to_categorical(y_train, 21)
    Y_test = np_utils.to_categorical(y_test, 21)
