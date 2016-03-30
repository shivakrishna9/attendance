import cv2
import pandas as pd
import numpy as np
import glob
import time
# 

def detect(image):

    img = cv2.imread(image)
    FACE_DETECTOR_PATH = "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    rects = detector.detectMultiScale(img, scaleFactor=1.4, minNeighbors=1,
                                      minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    
    for (x, y, w, h) in rects:
        roi_color = img[y:y + h, x:x + w]
        # cv2.imshow('image', roi_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    return roi_color

def categorize(x,n):
    array = np.zeros(n)
    array[x]=1
    # print array
    return array


def from_file(fname='classtrain.txt'):
    print 'Collecting images...'
    start = time.time()
    read = pd.read_csv(fname,names=['filename','name'])

    images=[]
    image_classes=[]
    for data in read.itertuples():
        image = data[1]
        print "Extracting", image
        image_class = data[2]
        image = glob.glob("newtest/*/"+image)[0]
        # if image_class not in image_classes:
        image_classes.append(image_class)
        image = cv2.imread(image)
        # print image.shape
        image = input_image(image)
        # print image.shape
        image = np.rollaxis(image,2,start=0)
        # print image.shape
        images.append(image)

    print "All images collected in ..",time.time() - start
    print "No. of images:", len([images][0]), "No. of classes:", len(image_classes)
    return np.array(images), image_classes

def test_file(fname='classtest.txt'):
    read = pd.read_csv(fname,names=['filename','name'])

    images=[]
    image_classes=[]
    for data in read.itertuples():
        image = data[1]
        image_class = data[2]
        image = glob.glob("newtest/*/"+image)[0]
        image_classes.append(categorize(image_class,7))
        image = cv2.imread(image)
        image = input_image(image)
        image = np.rollaxis(image,2,start=0)
        images.append(image)

    return np.array(images), image_classes


def input_image(image):

    # res = detect(image)
    res = cv2.resize(image, (227, 227), interpolation=cv2.INTER_CUBIC)

    return res