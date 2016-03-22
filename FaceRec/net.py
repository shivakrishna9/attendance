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
from FaceRec.pretrained_cnn import *
from keras import backend as K
import time


def VGGNet(X_train, y_train, X_test, Y_test):

    print "Initialising model..."
    start = time.time()
    model = Sequential()
    model.add(Convolution2D(64, 3, 3,input_shape=(3, 227, 227), activation='relu', name='conv1_1', border_mode='same'))
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
    model.add(Dense(output_dim=6, init="uniform"))
    model.add(Activation("softmax"))

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    print "model init in ..", time.time()-start

    print "Extracting pretrained data"
    start = time.time()
    cnn = pretrained_cnn()

    for k in cnn[cnn.keys()[0]]:
        for i in k:
            a = i[0][0][1][0]
            if 'conv' in a:
                weight1 = np.rollaxis(i[0][0][2][0][0],3,start=0)
                weight1 = np.rollaxis(weight1,3,start=1)
                weight2 = np.rollaxis(i[0][0][2][0][1],1,start=0)[0]
                weights = [weight1,weight2]
                layer_dict[a].set_weights(weights)
                print "Weights added to",a

    print "model extracted in ..",time.time()-start


    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    print "Compiling model..."
    start = time.time()
    model.compile(loss='mse', optimizer=adam)
    print "Model compiled in ..",time.time()-start

    # model.fit(X_train, y_train, nb_epoch=20, batch_size=1, verbose=1, show_accuracy=True, shuffle=True)
    model.train_on_batch(X_train, y_train, accuracy=True)
    model.save_weights("cnn_weights.h5",overwrite=True)

    objective_score = model.evaluate(X_test, Y_test, batch_size=16)

    print "Predicting for test images..."
    start = time.time()
    classes = model.predict_classes(X_test, batch_size=16)

    print objective_score
    print classes, Y_test
    print "Predicted in ..",time.time()-start
