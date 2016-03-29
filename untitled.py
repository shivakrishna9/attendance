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
import FaceRec
from FaceRec.net import *
from FaceRec.get_input import *
from scipy.misc import imsave

# util function to convert a tensor into a valid image
def visualise():
    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    img = input_img_data[0]
    img = deprocess_image(img)
    imsave('%s_filter_%d.png' % (layer_name, filter_index), img)

x,y = from_file()
x_test, y_test = test_file()

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

layer_dict = dict([(layer.name, layer) for layer in model.layers])

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

model.add(Flatten())
model.add(Dense(output_dim=4096, activation='relu', init="uniform"))
model.add(Dense(output_dim=4096, init="uniform", activation='relu'))
model.add(Dense(output_dim=7, init="uniform", activation='softmax'))



sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
for i in xrange(10):
    if (i+1)*16 > 87:
        break
    model.train_on_batch(x[i*16:(i+1)*16], y[(i)*16:(i+1)*16], accuracy=True)
    print model.evaluate(x_test, y_test, batch_size=4, show_accuracy=True)
    print model.predict(x_test, batch_size=4)

