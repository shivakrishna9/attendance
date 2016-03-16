import numpy as np
# import cv2
# import theano
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

    # with h5py.File('vgg_face.h5','w') as f:
    #   net = cnn['net'].tolist()
    #   f.create_dataset('vgg_face',data=net)
    return cnn

def import_weights(model,layer_dict):
    cnn = pretrained_cnn()

    for k in cnn[cnn.keys()[0]]:
        for i in k:
            # weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            a = i[0][0][1][0]
            if 'conv' in a:
                # print i[0][0][2][0][0].shape
                # print i[0][0][2][0][1].shape
                weight1 = np.rollaxis(i[0][0][2][0][0],3,start=0)
                weight1 = np.rollaxis(weight1,3,start=1)
                weight2 = np.rollaxis(i[0][0][2][0][1],1,start=0)[0]
                # print weight1.shape
                # print weight2.shape
                weights = [weight1,weight2]
                # print layer_dict[a].get_weights()[0].shape
                # print layer_dict[a].get_weights()[1].shape
                layer_dict[a].set_weights(weights)
                print "Weights added to",a
                # time.sleep(2)

        time.sleep(10)
        for i in k:
            if 'pool' in i[0][0][1][0]:
                print model.layers[5].get_weights()
                print i
