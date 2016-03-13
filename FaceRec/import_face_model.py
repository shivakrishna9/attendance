import sys
import caffe
import cv2
from PIL import Image
import numpy as np
from scipy.misc import imresize
caffe.set_mode_cpu()

home = "/home/walle/"

MODEL_FILE = home + 'models/vgg_face_caffe/VGG_FACE_deploy.prototxt'
PRETRAINED = home + 'models/vgg_face_caffe/VGG_FACE.caffemodel'

net = caffe.Net(MODEL_FILE, caffe.TEST)
net.params['conv1_1']