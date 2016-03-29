import numpy as np
import cv2
import json
import time
import numpy as np
import h5py
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

PRETRAINED = "cnn_weights.h5"
# relu = []
# PRETRAINED = "vgg-face.mat"

def pretrained_cnn():
    f = h5py.File(PRETRAINED)
    for k in range(f.attrs['nb_layers']):
        # if k >= len(model.layers):
        #     # we don't look at the last (fully-connected) layers in the savefile
        #     break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

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
