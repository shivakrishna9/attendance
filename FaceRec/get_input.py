import cv2
import pandas as pd
import glob

def detect(image):

    img = cv2.imread(image)
    FACE_DETECTOR_PATH = "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    rects = detector.detectMultiScale(img, scaleFactor=1.4, minNeighbors=1,
                                      minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = img[y:y + h, x:x + w]
    
    return roi_color


def from_file(fname='train.txt'):
    read = pd.read_csv(fname,names=['filename','name'])

    images=[]
    image_classes=[]
    for data in read.itertuples():
        image = data[1]
        image_class = data[2]
        image = glob.glob("newtest/*/"+image+".jpg")[0]
        # if image_class not in image_classes:
        image_classes.append(image_class)
        image = input_image(image)
        images.append(image)

    print "No. of images:", len(images), "No. of classes:", len(image_classes)
    return images, image_classes

def test_file(fname='test.txt'):
    read = pd.read_csv(fname,names=['filename','name'])

    images=[]
    image_classes=[]
    for data in read.itertuples():
        image = data[1]
        image_class = data[2]
        image = glob.glob("newtest/*/"+image+".jpg")[0]
        # if image_class not in image_classes:
        image_classes.append(image_class)
        image = input_image(image)
        images.append(image)

    return images, image_classes

def input_image(image):

    res = detect(image)
    res = cv2.resize(res, (227, 227), interpolation=cv2.INTER_CUBIC)

    return res