import numpy as np
import cv2
import theano
import scipy.io
import json
import time
import numpy as np
import h5py 
import tables
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

PRETRAINED = "extras/models/vgg-face.mat"
# relu = []
# PRETRAINED = "vgg-face.mat"

def pretrained_cnn():
    cnn = scipy.io.loadmat(PRETRAINED)
    return cnn

def import_weights(model,layer_dict):
    cnn = pretrained_cnn()

    # adding weights to the model    
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
