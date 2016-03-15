import numpy as np
# import cv2
# import theano
import scipy.io
import json

import numpy as np
import h5py 
import tables

# home = "/home/walle/"
PRETRAINED = "vgg_face_matconvnet/vgg_face.mat"

def pretrained_cnn():
	cnn = scipy.io.loadmat(PRETRAINED)

	with h5py.File('vgg_face.h5','w') as f:
		net = cnn['net'].tolist()
		f.create_dataset('vgg_face',data=net)

pretrained_cnn()
