import sys
import caffe
import cv2
from PIL import Image
import numpy as np
from scipy.misc import imresize
caffe.set_mode_cpu()

home = "/home/walle/"

MODEL_FILE = home + 'Attendance/extras/models/ResNet-101-deploy.prototxt'
PRETRAINED = home + 'Attendance/extras/models/ResNet-101-model.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TRAIN)

print net.params


