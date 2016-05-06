import numpy as np
import pandas as pd
import cv2
import keras
import h5py
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import scipy.io
from .get_input import *

NB_CLASS = 21
PRETRAINED = "extras/cnn_weights_class.h5"


def get_pt_mat(model, layer_dict):
    print "Extracting pretrained data"
    start = time.time()
    cnn = scipy.io.loadmat('extras/vgg-face.mat')

    for k in cnn[cnn.keys()[0]]:
        for i in k:
            a = i[0][0][1][0]
            # print a
            if 'conv' in a:
                weight1 = np.rollaxis(i[0][0][2][0][0], 3, start=0)
                weight1 = np.rollaxis(weight1, 3, start=1)
                weight2 = np.rollaxis(i[0][0][2][0][1], 1, start=0)[0]
                weights = [np.array(weight1).astype('float32'), np.array(weight2).astype('float32')]
                layer_dict[a].set_weights(weights)
                print "Weights added to", a

    print "model extracted in ..", time.time() - start

def evaluate(model, batch_size=16):
        print 'Evaluating, predicting and saving weights ..'

        chunks = pd.read_csv('traintest/class_20.txt',
            names=['person', 'class','image'], chunksize=1024, 
            sep='\t', engine='python')
        count = 0
        x = 0
        # print 'Epoch:',epoch,'/ 400'
        for data in chunks:
            X_train, y_train = class_db_read(data)
            batch = X_train.shape[0]
            count+=batch
            x += 1024
            print 'Count:',count, 'X:', x
            # if batch > 0:
            preds = model.predict(X_train, batch_size=batch_size)
            class_preds = np.argmax(preds, axis=1)
            cls = []
            for i,j in enumerate(class_preds):
                cls += [[j, np.max(preds[i])]]
            ev = model.evaluate(X_train,y_train, batch_size=4)

            print 'EVAL: '+str(ev)
            for i, j in enumerate(cls):
                print 'PRED: '+str(j), 'TRUE: '+str(np.argmax(y_train[i]))

            with open('outputs/class_eval.txt','a') as f:
                f.write('EVAL: '+str(ev)+'\n')
                for i, j in enumerate(cls):
                    f.write('PRED: '+str(j)+'\tTRUE: '+str(np.argmax(y_train[i]))+'\n')
        print 'Evaluated, predicted and saved weights !'

def epsw(model, batch_size=16):
        print 'Evaluating, predicting and saving weights ..'

        chunks = pd.read_csv('traintest/classtest.txt',
            names=['person', 'class','image'], chunksize=1024, 
            sep='\t', engine='python')
        count = 0
        x = 0
        # print 'Epoch:',epoch,'/ 400'
        for data in chunks:
            X_train, y_train = class_db_read(data)
            batch = X_train.shape[0]
            count+=batch
            x += 1024
            print 'Count:',count, 'X:', x
            # if batch > 0:
            preds = model.predict(X_train, batch_size=batch_size)
            # print preds
            class_preds = np.argmax(preds, axis=1)
            cls = []
            for i,j in enumerate(class_preds):
                cls += [[j, np.max(preds[i])]]
            # print cls
            # print np.argmax(y_train, axis=1)
            # print model.evaluate(X_train,y_train, batch_size=4)
            with open('outputs/class_preds.txt','a') as f:
                f.write('PREDS: '+str(cls)+'\n')
                f.write('TRUE: '+str(np.argmax(y_train, axis=1))+'\n')
        # model.save_weights("extras/cnn_weights_class.h5", overwrite=True)
        print 'Evaluated, predicted and saved weights !'


def train(model,batch_size=16, epochs=400, lr=1e-4, nb_epoch=1):
    print "Training on batch..."

    for epoch in xrange(0, epochs):
        start = time.time()
        chunks = pd.read_csv('traintest/train.txt',
            names=['person', 'class','image'], chunksize=256,
            sep='\t', engine='python')
        count = 0
        # for data in chunks:
        #     x_test, y_test = db_read(data)
        #     break
        x = 0
        print 'Epoch:',epoch,'/',epochs
        
        for data in chunks:
        
            X_train, y_train = class_db_read(data)
            print X_train.shape, y_train.shape
            batch = X_train.shape[0]
            count+=batch
            x += batch
            
            print 'Epoch:',epoch,'/ 400', 'Count:',count, 'X:', x, 'Learning Rate:',lr
        
            if batch > 0:
                model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
                               verbose=1, shuffle=True, validation_split=0.05)
                
                model.save_weights(PRETRAINED,overwrite=True)
        
        print "Total training time ..", time.time() - start
        epsw(model,batch_size=4)


def VGGNet(nb_epoch=1, batch_size=4):

    

    print "Initialising model..."
    start = time.time()
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, input_shape=(3, 227, 227),
                            activation='relu',trainable=False, name='conv1_1', border_mode='same'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',trainable=False, name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',trainable=False, name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',trainable=False, name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable=False, name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable=False, name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable=False, name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False, name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False, name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False, name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False, name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False, name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False, name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    print "model init in ..", time.time() - start

    print "Adding fully connected layers ..."
    start = time.time()
    model.add(Flatten())
    model.add(Dense(output_dim=4096, activation='relu',trainable=False, init="uniform"))
    model.add(Dense(output_dim=4096, init="uniform",trainable=False, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=NB_CLASS, init="uniform", activation='softmax'))
    print 'FC layers added ! Time taken :', time.time() - start

    print 'Loading weights ...'
    start = time.time()
    # get_pt_mat(model, layer_dict)
    model.load_weights(PRETRAINED)
    print 'Model loaded in ..', time.time() - start
    
    # model.save_weights(PRETRAINED,overwrite=True)
    lr = 1e-4
    sgd = SGD(lr=lr, decay=5e-4, momentum=0.9, nesterov=True)

    print "Compiling model..."
    start = time.time()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print "Model compiled in ..", time.time() - start

    # print 'Evaluating ..'
    # evaluate(model,batch_size=4)
    # print 'Checkout the results in class_eval.txt file !'

    train(model,batch_size=4,epochs=20,lr=lr,nb_epoch=1)

    
    
# images trained on: 241
# number of testing images: 63
# validationi images: 13