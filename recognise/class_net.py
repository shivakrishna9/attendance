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
from os.path import exists
import os
# from extrafiles import *

# people = ['abdul_karim', 'abdul_wajid', 'abhishek_bhatnagar', 'abhishek_joshi', 'aditya', 'ahsan', 'akshat', 'aly', 'aman', 'ameen', 'antriksh', 'anzal', 'ashar', 'asif', 'avishkar', 'bushra', 'chaitanya', 'dhawal', 'farhan', 'farheen', 'ghalib', 'habib', 'harsh', 'irfan_ansari', 'jeevan', 'manaff', 'manish', 'maria', 'mehrab', 'mohib', 'naeem', 'nikhil_mittal', 'nikhil_raman', 'not_face', 'prerit', 'raghib_ahsan', 'rahul', 'ravi', 'rehan', 'rezwan', 'rubab', 'sachin', 'sahil', 'saif', 'saifur', 'sajjad', 'sana', 'sapna', 'sarah_khan', 'sarah_masud', 'sarthak', 'shadab', 'shafiya', 'shahbaz', 'shahjahan', 'sharan', 'shivam', 'shoaib', 'shoib', 'shruti', 'suhani', 'sultana', 'sunny', 'sushmita', 'tushar', 'umar', 'zeya', 'zishan']

# NB_CLASS = len(people)



class VGG(object):
    NB_CLASS = 0
    people = []
    PRETRAINED = "extras/cnn_weights_class66.h5"

    def demo(self,imgs,X_train=None,batch_size=2):

        preds = self.model.predict(X_train, batch_size=batch_size)

        list_best5 = []
        for i, j in enumerate(preds):
            prob = j
            prob5 = sorted(prob, reverse=True)[:5]
            best5 = []
            for k in prob5:
                best5.append([k, np.where(j == k)])

            list_best5.append(best5)

        # evl = self.model.evaluate(X_train, y_train, batch_size=4)
        # print evl
        for i, j in enumerate(list_best5):
            best5 = j
            print best5
            for k in best5:
                print self.people[k[1][0]], k[0]

            if best5[0][0]<0.30:
                t = (self.people[best5[0][1][0]])
                print 'First prediction:', t
                print 'due to low confidence, this image has not been attended to'
            if best5[0][0]>0.30 and best5[0][0]<0.70 :
                t = (self.people[j[0][1][0]])
                print 'First prediction:', t
                print 'due to medium confidence, I would like you to check this image'
                cv2.imshow(str(t), input_image(imgs[i]))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if best5[0][0]>0.70:
                t = (self.people[j[0][1][0]])
                print 'First prediction:', t
                print 'I have confidence on my prediction !'
                cv2.imshow(str(t), input_image(imgs[i]))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        people = []
        for pred in preds:
            if np.max(pred)>0.30:
                people.append(self.people[np.argmax(pred)])

        print people

        return people


    def run(self, frame, batch_size=4):

        X_train, imgs = detect_haar(frame, NB_CLASS=len(self.people))
        # print X_train.shape

        low = []
        mid = []
        high = []

        if X_train.any():

            preds = self.model.predict(X_train, batch_size=batch_size)

            list_best5 = []
            for i, j in enumerate(preds):
                prob = j
                prob5 = sorted(prob, reverse=True)[:5]
                best5 = []
                for k in prob5:
                    best5.append([k, np.where(j == k)])

                list_best5.append(best5)
                
            for i, j in enumerate(list_best5):
                best5 = j
                print best5
                if best5[0][0] < 0.30:
                    t = self.people[best5[0][1][0]]
                    print 'First prediction:', t
                    print 'due to low confidence, this image has not been attended to'
                    low.append(t)
                if best5[0][0] > 0.30 and best5[0][0] < 0.70:
                    t = self.people[j[0][1][0]]
                    print 'First prediction:', t
                    print 'due to medium confidence, I would like you to check this image'
                    mid.append(t)
                if best5[0][0] > 0.70:
                    t = self.people[j[0][1][0]]
                    print 'First prediction:', t
                    print 'I have confidence on my prediction !'
                    high.append(t)

        return low, mid, high


    def get_pt_mat(self,layer_dict):
        print "Extracting self.pretrained data"
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
                    weights = [np.array(weight1).astype(
                        'float32'), np.array(weight2).astype('float32')]
                    layer_dict[a].set_weights(weights)
                    print "Weights added to", a

        print "self.model extracted in ..", time.time() - start


    def evaluate(self, batch_size=4):
        print 'Evaluating, predicting and saving weights ..'

        chunks = pd.read_csv('traintest/demo.txt',
                             names=['person', 'class', 'image'], chunksize=256,
                             sep='\t', engine='python')
        # self.people = pre_process()
        count = 0
        x = 0
        for data in chunks:
            imgs, s = class_db_read1(data)
            X_train, y_train = s
            batch = X_train.shape[0]
            count += batch
            print 'Count:', count, 'X:', x
            preds = self.model.predict(X_train, batch_size=batch_size)

            list_best5 = []
            for i, j in enumerate(preds):
                prob = j
                prob5 = sorted(prob, reverse=True)[:5]
                best5 = []
                for k in prob5:
                    best5.append([k, np.where(j == k)])

                list_best5.append(best5)

            # class_preds = np.argmax(preds, axis=1)
            # cls = []
            # for i, j in enumerate(class_preds):
            #     cls += [[j, np.max(preds[i])]]
            # print cls
            # print np.argmax(y_train, axis=1)
            evl = self.model.evaluate(X_train, y_train, batch_size=4)
            print evl
            for i, j in enumerate(list_best5):
                # print np.argmax(y_train[i])
                best5 = j
                for k in best5:
                    # print k[1][0], k[0]
                    print self.people[k[1][0]], k[0]

                t = (self.people[j[0][1][0]], self.people[np.argmax(y_train[i])])
                print 'First prediction:', t

                # cv2.imwrite('outputs/classification/'+str(t)+'.jpg', cv2.imread(imgs[i]))

                cv2.imshow(str(t), cv2.imread(imgs[i]))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print 'Evaluated, predicted and saved weights !'


    def epsw(self,batch_size=16):
        print 'Evaluating, predicting and saving weights ..'

        chunks = pd.read_csv('traintest/class20_test.txt',
                             names=['person', 'class', 'image'], chunksize=1024,
                             sep='\t', engine='python')
        count = 0
        x = 0
        for data in chunks:
            X_train, y_train = class_db_read(data)
            batch = X_train.shape[0]
            count += batch
            x += 1024
            print 'Count:', count, 'X:', x
            preds = self.model.predict(X_train, batch_size=batch_size)
            class_preds = np.argmax(preds, axis=1)
            cls = []
            for i, j in enumerate(class_preds):
                cls += [[j, np.max(preds[i])]]
            # print cls
            # print np.argmax(y_train, axis=1)
            evl = self.model.evaluate(X_train, y_train, batch_size=4)
            print evl
            with open('outputs/class20_preds.txt', 'a') as f:
                f.write('EVAL: ' + str(evl) + '\n')
                for i, j in enumerate(cls):
                    f.write('PRED: ' + str(j) + '\tTRUE: ' +
                            str(np.argmax(y_train[i])) + '\n')
        # self.model.save_weights("extras/cnn_weights_class.h5", overwrite=True)
        print 'Evaluated, predicted and saved weights !'


    def train(self, batch_size=2, epochs=400, lr=2.5e-4, nb_epoch=1):
        print "Training on batch..."

        for epoch in xrange(0, epochs):
            start = time.time()
            chunks = pd.read_csv('traintest/class66_train.txt',
                                 names=['person', 'class', 'image'], chunksize=256,
                                 sep='\t', engine='python')
            count = 0
            # for data in chunks:
            #     x_test, y_test = db_read(data)
            #     break
            x = 0
            print 'Epoch:', epoch, '/', epochs

            for data in chunks:

                X_train, y_train = class_db_read(data)
                print X_train.shape, y_train.shape
                batch = X_train.shape[0]
                count += batch
                x += batch

                print 'Epoch:', epoch, '/', epochs, 'Count:', count, 'X:', x, 'Learning Rate:', lr

                if batch > 0:
                    self.model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
                              verbose=1, shuffle=True)

                    self.model.save_weights(self.PRETRAINED, overwrite=True)

            print "Total training time ..", time.time() - start
            epsw(self.model, batch_size=4)


    def VGGNet(self,plist,NB_CLASS=NB_CLASS, nb_epoch=1, batch_size=4):
        self.people=plist
        self.NB_CLASS = NB_CLASS
        self.PRETRAINED = 'extras/cnn_weights_class'+(NB_CLASS-2)+'.h5'

        print "Initialising self.model..."
        start = time.time()
        self.model = Sequential()
        self.model.add(Convolution2D(64, 3, 3, input_shape=(3, 227, 227),
                                activation='relu', trainable=False, name='conv1_1', border_mode='same'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu',
                                trainable=False, name='conv1_2'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu',
                                trainable=False, name='conv2_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, 3, 3, activation='relu',
                                trainable=False, name='conv2_2'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu',
                                trainable=False, name='conv3_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu',
                                trainable=False, name='conv3_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, 3, 3, activation='relu',
                                trainable=False, name='conv3_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu',
                                trainable=False, name='conv4_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu',
                                trainable=False, name='conv4_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu',
                                trainable=False, name='conv4_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu',
                                trainable=False, name='conv5_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu',
                                trainable=False, name='conv5_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, 3, 3, activation='relu',
                                trainable=False, name='conv5_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        print "self.model init in ..", time.time() - start

        if self.PRETRAINED:

            print "Adding fully connected layers ..."
            start = time.time()
            self.model.add(Flatten())
            self.model.add(Dense(output_dim=4096, activation='relu',
                            trainable=False, init="uniform"))
            self.model.add(Dense(output_dim=4096, init="uniform",
                            activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(output_dim=self.NB_CLASS, init="uniform", activation='softmax'))
            print 'FC layers added ! Time taken :', time.time() - start

            print 'Loading weights ...'
            start = time.time()
            # get_pt_mat(self.model, layer_dict)
            self.model.load_weights(self.PRETRAINED)
            print 'self.Model loaded in ..', time.time() - start

        else:
            get_pt_mat(self.model, layer_dict)

            print "Adding fully connected layers ..."
            start = time.time()
            self.model.add(Flatten())
            self.model.add(Dense(output_dim=4096, activation='relu',
                            trainable=False, init="uniform"))
            self.model.add(Dense(output_dim=4096, init="uniform",
                            activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(output_dim=self.NB_CLASS, init="uniform", activation='softmax'))
            print 'FC layers added ! Time taken :', time.time() - start

        # self.model.save_weights(self.PRETRAINED,overwrite=True)
        lr = 1.5e-4
        sgd = SGD(lr=lr, decay=5e-4, momentum=0.9, nesterov=True)

        print "Compiling self.model..."
        start = time.time()
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])
        print "self.Model compiled in ..", time.time() - start

        # print 'Evaluating ..'
        # self.evaluate(self.model,  batch_size=4)
        # print 'Checkout the results in class_eval.txt file !'

        # self.train(self.model, batch_size=4, epochs=400, lr=lr, nb_epoch=1)


    # images trained on: 241
    # number of testing images: 63
    # validationi images: 13

    