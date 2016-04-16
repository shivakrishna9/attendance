import numpy as np
import cv2
import keras
from keras.models import Sequential, Graph, Model
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
from keras.layers import BatchNormalization


def resnet101():

	conv1 = Sequential()
    conv1.add(ZeroPadding2D((3, 3)))
    conv1.add(Convolution2D(64, 7, 7, subsample=(2,2), name='conv', border_mode='same'))
    conv1.add(BatchNormalization(), name='bn')
    conv1.add(Activation('relu'), name='relu')

    res2a_branch1 = Sequential()
    res2a_branch1.add(Convolution2D(256, 1, 1, name='conv'))
    res2a_branch1.add(BatchNormalization(), name='bn')

    res2a_branch2a = Sequential()
    res2a_branch2a.add(Convolution2D(64, 1, 1, name='conv'))
    res2a_branch2a.add(BatchNormalization(), name='bn')
    res2a_branch2a.add(Activation('relu'), name='relu')

    res2a_branch2b = Sequential()
    res2a_branch2b.add(ZeroPadding2D((1, 1)))
    res2a_branch2b.add(Convolution2D(64, 3, 3, name='conv'))
    res2a_branch2b.add(BatchNormalization(), name='bn')
    res2a_branch2b.add(Activation('relu'), name='relu')

    res2a_branch2c = Sequential()
    res2a_branch2c.add(Convolution2D(256, 1, 1, name='conv'))
    res2a_branch2c.add(BatchNormalization(), name='bn')

    # res2a = Sequential()
    # res2a.add(Convolution2D(256, 1, 1, name='conv'))

    # rpn_graph = Graph()
    # rpn_graph.add_input(name='RPNinput', input_shape=(3, 227, 227))
    # for i in xrange(9):
    #     graph.add_node(Convolution2D(512, 3, 3, activation='relu'), name='rpn'+i, input='input1')    


	# graph = Model()
	input1 = Input(input_shape=(3, 227, 227))
	conv1 = Model(conv1)(input1)

    pool1 = graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)))(conv1)

    res2a_branch1 = Model(res2a_branch1)(pool1)
    res2a_branch2a = Model(res2a_branch2a)(pool1)
    res2a_branch2b = Model(res2a_branch2b, name='', input='res2a_branch2a')
    graph.add_node(res2a_branch2c, name='res2a_branch2c', input='res2a_branch2b')

    x = merge(input=[res2a_branch2b, res2a_branch1], mode='sum')

    res2a_relu = Model(Activation('relu'))(x)

    res2b_branch2a = Model(res2a_branch2a)(res2a_relu)
    graph.add_node(res2a_branch2b, name='res2b_branch2b', input='res2b_branch2a')
    graph.add_node(res2a_branch2c, name='res2b_branch2c', input='res2a_branch2b')

    x = merge(input=[res2a_branch2b, res2a_branch1], mode='sum')

    res2a_relu = Model(Activation('relu'), name='res2a_relu')(x)





    # RPN Model to be added
    graph.add_node(Flatten(), input='cnn', name='flatten1')
    graph.add_node(Dense(output_dim=4096, activation='relu'), name='dense1', input='flatten1')
    graph.add_node(Dense(output_dim=4096, activation='relu'), name='dense2', input='dense1')
    graph.add_node(Dropout(0.5), name='dropout1', input='dense2')
    graph.add_node(Dense(output_dim=21, activation='softmax'), name='denseOut', input='dropout1')
    graph.add_output(name='output', input='denseOut')


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
