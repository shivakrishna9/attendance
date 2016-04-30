import numpy as np
import cv2
import keras
import h5py
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

NB_CLASS = 21


def get_pt_mat():
    print "Extracting pretrained data"
    start = time.time()
    cnn = pretrained_cnn()

    for k in cnn[cnn.keys()[0]]:
        for i in k:
            a = i[0][0][1][0]
            if 'conv' in a:
                weight1 = np.rollaxis(i[0][0][2][0][0], 3, start=0)
                weight1 = np.rollaxis(weight1, 3, start=1)
                weight2 = np.rollaxis(i[0][0][2][0][1], 1, start=0)[0]
                weights = [weight1, weight2]
                layer_dict[a].set_weights(weights)
                print "Weights added to", a

    print "model extracted in ..", time.time() - start

    print "Saving weights..."
    model.save_weights("cnn_weights.h5", overwrite=True)
    print "Batch trained and weights saved !"


def VGGNet(X_train, y_train, X_test, Y_test):

    PRETRAINED = "extras/cnn_weights.h5"

    print "Initialising model..."
    start = time.time()
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

    node_dict = [(node) for node in model.layers]
    print "model init in ..", time.time() - start

    print "Adding fully connected layers ..."
    start = time.time()
    model.add(Flatten())
    model.add(Dense(output_dim=4096, activation='relu', init="uniform"))
    model.add(Dense(output_dim=4096, init="uniform", activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=NB_CLASS, init="uniform", activation='softmax'))
    print 'FC layers added ! Time taken :', time.time() - start

    print 'Loading weights ...'
    start = time.time()
    model.load_weights(PRETRAINED)

    # f = h5py.File(PRETRAINED)
    # for k in range(f.attrs['nb_layers']):
    #     if k >= len(model.layers):
    #         break
    #     g = f['layer_{}'.format(k)]
    #     weights = [g['param_{}'.format(p)]
    #                for p in range(g.attrs['nb_params'])]
    #     model.layers[k].set_weights(weights)
    # f.close()
    print 'Model loaded in ..', time.time() - start

    sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)

    print "Compiling model..."
    start = time.time()
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print "Model compiled in ..", time.time() - start

    print 'Normalizing data...'
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train = X_train - np.average(X_train)
    X_test = X_test - np.average(X_test)
    y_train = np_utils.to_categorical(y_train, NB_CLASS)
    Y_test = np_utils.to_categorical(Y_test, NB_CLASS)

    print "Training on batch..."
    start = time.time()
    for i in xrange(10):
        model.fit(X_train, y_train, nb_epoch=2, batch_size=16,
                  verbose=1, show_accuracy=True, shuffle=True)
        print model.evaluate(X_test, Y_test, batch_size=4, show_accuracy=True)
        print model.predict(X_test, batch_size=4)
    print "Total training time ..", time.time() - start, "Saving weights..."
    
    # model.save_weights("cnn_weights.h5",overwrite=True)
    # print "Batch trained and weights saved !"

    objective_score = model.evaluate(
        X_test, Y_test, batch_size=4, show_accuracy=True)

    print "Predicting for test images..."
    start = time.time()
    classes = model.predict(X_test, batch_size=4)

    print objective_score
    print classes
    print Y_test
    print "Predicted in ..", time.time() - start
