import numpy as np
import cv2
import theano
import scipy.io

home = "/home/walle/"
PRETRAINED = home + "models/vgg-face.mat"

def pretrained_cnn():
	cnn = scipy.io.loadmat(PRETRAINED)['layers']
	print cnn
