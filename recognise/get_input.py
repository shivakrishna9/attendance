import cv2
import pandas as pd
import numpy as np
import glob
import time
from keras.utils import np_utils
im_size = 227

def detect(image,dets):

    # img = cv2.imread(image)
    # FACE_DETECTOR_PATH = "extras/haarcascade_frontalface_default.xml"

    # detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    # rects = detector.detectMultiScale(img, scaleFactor=1.4, minNeighbors=1,
    #                                   minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    # print image
    # print dets.split(',')
    x, y, w, h = dets.split(',')
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    if x!=0 and y!=0 and w!=0 and h!=0:
        roi_color = image[y:y + h, x:x + w]
    else:
        roi_color = image

    return roi_color


def categorize(x, n):
    array = np.zeros(n)
    array[x] = 1
    # print array
    return array


def from_file(fname='traintest/train.txt'):
    print 'Collecting training images...'
    start = time.time()
    read = pd.read_csv(fname, names=['filename', 'name'])

    images = []
    image_classes = []
    for data in read.itertuples():
        image = data[1]
        image_class = data[2]
        # print "extracting", image, image_class
        image = glob.glob("newtest/*/" + image)
        # print image
        image_classes.append(image_class)
        image = cv2.imread(image[0])
        image = input_image(image)
        image = np.rollaxis(image, 2, start=0)
        images.append(image)

    print "Training images collected in ..", time.time() - start
    # print "No. of images:", len([images][0]), "No. of classes:",
    # len(image_classes)
    return np.array(images).astype('float32'), image_classes


def test_file(fname='traintest/classtest.txt'):
    print 'Collecting testing images...'
    start = time.time()
    read = pd.read_csv(fname, names=['filename', 'name'])

    images = []
    image_classes = []
    for data in read.itertuples():
        image = data[1]
        image_class = data[2]
        image = glob.glob("newtest/*/" + image)[0]
        image_classes.append(image_class)
        image = cv2.imread(image)
        image = input_image(image)
        image = np.rollaxis(image, 2, start=0)
        images.append(image)

    print "Testing images collected in ..", time.time() - start
    return np.array(images).astype('float32'), image_classes


def input_image(image):

    # res = detect(image)
    try:
        res = cv2.resize(image, (im_size, im_size), interpolation=cv2.INTER_CUBIC)
    except cv2.error:
        res = None

    return res

def preprocess(images, classes):

    NB_CLASS = 696
    images = images.astype('float32')
    images /= 255
    images = images - np.average(images)
    classes = np_utils.to_categorical(classes, NB_CLASS)
    # dets = dets.astype('float32')

    return images, classes


def db_read(chunk):
    images = []
    image_classes = []
    image_dets = []
    for data in chunk.itertuples():
        # print data
        image = data[3]
        image_det = data[4]
        person = data[1]
        image_class = data[2]

        if cv2.imread(image)!=None:
            image = cv2.imread(image)
            image = input_image(detect(image,image_det))
            
            if image==None:
                continue
            else:
                image = np.rollaxis(image, 2, start=0)
                images.append(image)
                image_classes.append(image_class)
        else:
            # print image, 'image not found !'
            continue

    return preprocess(np.array(images), np.array(image_classes))
