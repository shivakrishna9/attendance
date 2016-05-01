import numpy as np
import cv2
import keras
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout, Activation
from keras.layers import Input, merge
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD, Adam
import h5py
import time
from keras.utils import np_utils
from recognise.get_input import *
import tensorflow as tf
# from rpn.anchor_target_layer import AnchorTargetLayer

NB_CLASS = 696  # number of classes
# 'th' (channels, width, height) or 'tf' (width, height, channels)
DIM_ORDERING = 'th'
WEIGHT_DECAY = 0.  # L2 regularization factor
USE_BN = True  # whether to use batch normalization

# import memory_profiler

# @profile
class ResNet():

    def get_data(self):
        self.X_train, self.y_train = imdb()

    # def get_pt(self):

    #     home = "/home/walle/"

    #     MODEL_FILE = home + 'Attendance/extras/models/ResNet-101-deploy.prototxt'
    #     PRETRAINED = home + 'Attendance/extras/models/ResNet-101-model.caffemodel'

    #     net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TRAIN)

    #     params = net.params.keys()

    #     self.source_params = {pr: (net.params[pr][0].data) for pr in params}

        # caffe.reset_all()

    def conv2D_bn_relu(self, x, nb_filter, nb_row, nb_col,
                       border_mode='valid', subsample=(1, 1),
                       activation='relu', batch_norm=USE_BN,
                       padding=(0, 0), weight_decay=WEIGHT_DECAY,
                       dim_ordering=DIM_ORDERING, name=None):
        '''Utility function to apply to a tensor a module conv + BN + ReLU
        with optional weight decay (L2 weight regularization).
        '''
        if weight_decay:
            W_regularizer = regularizers.l2(weight_decay)
            b_regularizer = regularizers.l2(weight_decay)
        else:
            W_regularizer = None
            b_regularizer = None

        if padding != (0, 0):
            x = ZeroPadding2D(padding)(x)
        x = Convolution2D(nb_filter, nb_row, nb_col,
                          subsample=subsample,
                          border_mode=border_mode,
                          W_regularizer=W_regularizer,
                          b_regularizer=b_regularizer,
                          dim_ordering=DIM_ORDERING,
                          name=name)(x)
        if batch_norm:
            if name == 'conv1':
                bn_name = 'bn_' + name
            else:
                bn_name = 'scale' + name.replace('res', '')
            x = BatchNormalization(name=bn_name)(x)
        if activation == 'relu':
            x = Activation('relu')(x)
        return x

    def resnet101(self):

        # ResNet-101 using functional API from keras
        # with tf.device('/cpu:0'):    
        print 'Initialising ResNet-101 !'
        start = time.time()
        if DIM_ORDERING == 'th':
            input1 = Input(shape=(3, 227, 227))
            CONCAT_AXIS = 1
        elif DIM_ORDERING == 'tf':
            input1 = Input(shape=(227, 227, 3))
            CONCAT_AXIS = 3
        else:
            raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

        conv1 = self.conv2D_bn_relu(
            input1, 64, 7, 7, name='conv1', padding=(3, 3), subsample=(2, 2))

        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

        res2a_branch1 = self.conv2D_bn_relu(
            pool1, 256, 1, 1, name='res2a_branch1')
        res2a_branch2a = self.conv2D_bn_relu(
            pool1, 64, 1, 1, name='res2a_branch2a')
        res2a_branch2b = self.conv2D_bn_relu(
            res2a_branch2a, 64, 3, 3, name='res2a_branch2b', padding=(1, 1))
        res2a_branch2c = self.conv2D_bn_relu(
            res2a_branch2b, 256, 1, 1, name='res2a_branch2c')

        x = merge([res2a_branch2c, res2a_branch1], mode='sum')

        res2a_relu = Activation('relu')(x)

        res2b_branch2a = self.conv2D_bn_relu(
            res2a_relu, 64, 1, 1, name='res2b_branch2a')
        res2b_branch2b = self.conv2D_bn_relu(
            res2b_branch2a, 64, 3, 3, name='res2b_branch2b', padding=(1, 1))
        res2b_branch2c = self.conv2D_bn_relu(
            res2b_branch2b, 256, 1, 1, name='res2b_branch2c')

        x = merge([res2a_relu, res2b_branch2c], mode='sum')

        res2b_relu = Activation('relu')(x)

        res2c_branch2a = self.conv2D_bn_relu(
            res2b_relu, 64, 1, 1, name='res2c_branch2a')
        res2c_branch2b = self.conv2D_bn_relu(
            res2c_branch2a, 64, 3, 3, name='res2c_branch2b', padding=(1, 1))
        res2c_branch2c = self.conv2D_bn_relu(
            res2c_branch2b, 256, 1, 1, name='res2c_branch2c')

        x = merge([res2b_relu, res2c_branch2c], mode='sum')

        res2c_relu = Activation('relu')(x)

        res3a_branch1 = self.conv2D_bn_relu(
            res2c_relu, 512, 2, 2, name='res3a_branch1', subsample=(2, 2))
        res3a_branch2a = self.conv2D_bn_relu(
            res2c_relu, 128, 2, 2, name='res3a_branch2a', subsample=(2, 2))
        res3a_branch2b = self.conv2D_bn_relu(
            res3a_branch2a, 128, 3, 3, name='res3a_branch2b', padding=(1, 1))
        res3a_branch2c = self.conv2D_bn_relu(
            res3a_branch2b, 512, 1, 1, name='res3a_branch2c')

        x = merge([res3a_branch2c, res3a_branch1], mode='sum')

        for i in xrange(3):
            res3b_relu = Activation('relu')(x)
            res3b_branch2a = self.conv2D_bn_relu(
                res3b_relu, 128, 1, 1, name='res3b{}_branch2a'.format(i + 1))
            res3b_branch2b = self.conv2D_bn_relu(
                res3a_branch2a, 128, 3, 3, name='res3b{}_branch2b'.format(i + 1), padding=(1, 1))
            res3b_branch2c = self.conv2D_bn_relu(
                res3a_branch2b, 512, 1, 1, name='res3b{}_branch2c'.format(i + 1))

            x = merge([res3b_branch2c, res3b_relu], mode='sum')

        res3b3_relu = Activation('relu')(x)

        res4a_branch1 = self.conv2D_bn_relu(
            res3b3_relu, 1024, 2, 2, name='res4a_branch1', subsample=(2, 2))
        res4a_branch2a = self.conv2D_bn_relu(
            res3b3_relu, 256, 2, 2, name='res4a_branch2a', subsample=(2, 2))
        res4a_branch2b = self.conv2D_bn_relu(
            res4a_branch2a, 256, 3, 3, name='res4a_branch2b', padding=(1, 1))
        res4a_branch2c = self.conv2D_bn_relu(
            res4a_branch2b, 1024, 1, 1, name='res4a_branch2c')

        x = merge([res4a_branch2c, res4a_branch1], mode='sum')

        for i in xrange(22):
            res4b_relu = Activation('relu')(x)

            res4b_branch2a = self.conv2D_bn_relu(
                res4b_relu, 256, 1, 1, name='res4b{}_branch2a'.format(i + 1))
            res4b_branch2b = self.conv2D_bn_relu(
                res4b_branch2a, 256, 3, 3, name='res4b{}_branch2b'.format(i + 1), padding=(1, 1))
            res4b_branch2c = self.conv2D_bn_relu(
                res4b_branch2b, 1024, 1, 1, name='res4b{}_branch2c'.format(i + 1))

            x = merge([res4b_branch2c, res4b_relu], mode='sum')

        res4b22_relu = Activation('relu')(x)

        res5a_branch1 = self.conv2D_bn_relu(
            res4b22_relu, 2048, 2, 2, name='res5a_branch1', subsample=(2, 2))
        res5a_branch2a = self.conv2D_bn_relu(
            res4b22_relu, 512, 2, 2, name='res5a_branch2a', subsample=(2, 2))
        res5a_branch2b = self.conv2D_bn_relu(
            res5a_branch2a, 512, 3, 3, name='res5a_branch2b', padding=(1, 1))
        res5a_branch2c = self.conv2D_bn_relu(
            res5a_branch2b, 2048, 1, 1, name='res5a_branch2c')

        x = merge([res5a_branch2c, res5a_branch1], mode='sum')

        res5a_relu = Activation('relu')(x)

        res5b_branch2a = self.conv2D_bn_relu(
            res5a_relu, 512, 1, 1, name='res5b_branch2a')
        res5b_branch2b = self.conv2D_bn_relu(
            res5b_branch2a, 512, 3, 3, name='res5b_branch2b', padding=(1, 1))
        res5b_branch2c = self.conv2D_bn_relu(
            res5b_branch2b, 2048, 1, 1, name='res5b_branch2c')

        x = merge([res5a_relu, res5b_branch2c], mode='sum')

        res5b_relu = Activation('relu')(x)

        res5c_branch2a = self.conv2D_bn_relu(
            res5b_relu, 512, 1, 1, name='res5c_branch2a')
        res5c_branch2b = self.conv2D_bn_relu(
            res5c_branch2a, 512, 3, 3, name='res5c_branch2b', padding=(1, 1))
        res5c_branch2c = self.conv2D_bn_relu(
            res5c_branch2b, 2048, 1, 1, name='res5c_branch2c')

        x = merge([res5b_relu, res5c_branch2c], mode='sum')

        res5c_relu = Activation('relu')(x)

        # Region Proposal Network here
        pool5 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(res5c_relu)

        x = Dropout(0.5)(pool5)
        x = Flatten()(x)
        final = Dense(NB_CLASS, activation='softmax')(x)
        self.model = Model(input1, output=[final])

        print 'Initialised fast_rcnn in ..', time.time() - start
        # return preds, auxpreds, input1

    def init_weights(self):
        print 'Initialising weights from pretrained model ...'

        for layer in model.layers:
            print layer.name
            layer.set_weights()

        print 'Weights initialised !'

    def compile_net(self, optimizer='sgd'):

        # self.model.load_weights('extras/resnet_weights.h5')
        optimizer = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)

        print "Compiling model with nesterov momentum .."
        start = time.time()

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print 'Compiled in ..', time.time() - start

    def train_net(self, nb_epoch=2, batch_size=4):
        print "Training on batch..."
        start = time.time()

        for epoch in xrange(400):
            chunks = pd.read_csv('traintest/training2.txt',
                names=['person', 'class','image', 'bbox'], chunksize=256, 
                sep='\t', engine='python')
            count = 0
            x = 0
            print 'Epoch:',epoch,'/ 400'
            for data in chunks:
                self.X_train, self.y_train = imdb_read(data)
                # print self.X_train.shape
                # print self.y_train.shape
                # break
                batch = self.X_train.shape[0]
                print batch
                count+=batch
                x += batch
                if batch > 0:
                    self.model.fit(self.X_train, self.y_train, nb_epoch=nb_epoch, batch_size=batch_size,
                                   verbose=1, shuffle=True)
                    if x>=256:
                        x=0
                        self.epsw(batch=4)

            self.epsw(batch=4)

        print "Total training time ..", time.time() - start

    def epsw(self, batch=16):
        print 'Evaluating, predicting and saving weights ..'

        # print self.model.evaluate(self.X_test, self.Y_test, batch_size=batch)
        # preds = self.model.predict(self.X_test, batch_size=batch)
        self.model.save_weights("extras/resnet_weights.h5", overwrite=True)

        # print preds
        # print np.argmax(preds, axis=1)
        # print np.argmax(self.Y_test, axis=1)

        print 'Evaluated, predicted and saved weights !'

    # def normalise_data(self):
    #     print 'Normalizing data...'
    #     # self.X_train = self.X_train.astype('float32')
    #     # self.X_test = self.X_test.astype('float32')
    #     self.X_train /= 255
    #     self.X_test /= 255
    #     self.X_train = self.X_train - np.average(self.X_train)
    #     self.X_test = self.X_test - np.average(self.X_test)
    #     self.y_train = np_utils.to_categorical(self.y_train, NB_CLASS)
    #     self.Y_test = np_utils.to_categorical(self.Y_test, NB_CLASS)        
