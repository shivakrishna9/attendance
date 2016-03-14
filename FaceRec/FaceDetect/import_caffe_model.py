import sys
import caffe
import cv2
from PIL import Image
import numpy as np
from scipy.misc import imresize
caffe.set_mode_cpu()

home = "/home/walle/"

MODEL_FILE = 'deploy.prototxt'
PRETRAINED = home + 'models/coco_vgg16_faster_rcnn_final.caffemodel'

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TRAIN)
