import cv2
import pandas as pd
import numpy as np
import glob
import time
from keras.utils import np_utils
im_size = 227


def detect_haar(image):

    FACE_DETECTOR_PATH = "../extras/haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    rects = detector.detectMultiScale(img, scaleFactor=1.03, minNeighbors=5,
                                      minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    faces = []
    for (x, y, w, h) in rects:
        faces += [image[y:y + h, x:x + w]]

    return faces


def detect(image, dets):

    x, y, w, h = dets.split(',')
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    if x != 0 and y != 0 and w != 0 and h != 0:
        roi_color = image[y:h, x:w]
    else:
        roi_color = image

    # print roi_color

    # cv2.imshow('image', roi_color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return roi_color


def categorize(x, n):
    array = np.zeros(n)
    array[x] = 1
    # print array
    return array

def class_db_read1(chunk):

    images = []
    image_classes = []
    imgs = []
    for data in chunk.itertuples():
        image = data[3]
        imgs.append(image)
        person = data[1]
        image_class = data[2]
        # print image, person, image_class
        image = cv2.imread(image)
        image = input_image(image)
        image = np.rollaxis(image, 2, start=0)
        images.append(image)
        image_classes.append(image_class)

    return imgs, preprocess(np.array(images), np.array(image_classes), NB_CLASS=67)


def class_db_read(chunk):

    images = []
    image_classes = []

    for data in chunk.itertuples():
        image = data[3]
        person = data[1]
        image_class = data[2]
        # print image, person, image_class
        image = cv2.imread(image)
        image = input_image(image)
        image = np.rollaxis(image, 2, start=0)
        images.append(image)
        image_classes.append(image_class)

    return preprocess(np.array(images), np.array(image_classes), NB_CLASS=67)


def input_image(image):

    try:
        res = cv2.resize(image, (im_size, im_size),
                         interpolation=cv2.INTER_CUBIC)
    except cv2.error:
        res = None

    return res


def preprocess(images, classes, NB_CLASS=696):

    # NB_CLASS = 696
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
        if cv2.imread(image) != None:
            image = cv2.imread(image)
            image = input_image(detect(image, image_det))

            if image == None:
                continue
            else:
                image = np.rollaxis(image, 2, start=0)
                images.append(image)
                image_classes.append(image_class)
        else:
            # print image, 'image not found !'
            continue

    return preprocess(np.array(images), np.array(image_classes), NB_CLASS=696)
