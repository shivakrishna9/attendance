import numpy as np
import cv2
import theano
# import theano.tensor as T
# from theano.tensor.signal import conv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
import h5py
import time
import glob
from keras.preprocessing.text import one_hot


def one_hot_names():
    lst = []
    with open("lfwpeople.txt",'r') as f:
        for line in f:
            lst += [line.split(',')[1]]
    print one_hot(",".join(lst), 5749, split=",")[:10]
    print lst[:10]


def preprocess():
    
    with open("lfwpeople.txt",'w') as f:
        for image in glob.glob("extras/lfw/*/*.jpg"):
            print image.split('/')[3].split('.')[0]+","+image.split('/')[2]
            f.write(image.split('/')[3].split('.')[0]+","+image.split('/')[2]+'\n')



def image():
    # Load an color image in grayscale

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # print img.shape
    for image in glob.glob("newtest/*/*.jpg"):
        start = time.time()
        img = cv2.imread(image)
        # res = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)

        FACE_DETECTOR_PATH = "haarcascade_frontalface_default.xml"

        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        rects = detector.detectMultiScale(img, scaleFactor=1.4, minNeighbors=1,
                                          minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # construct a list of bounding boxes from the detection
        # rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

        # update the data dictionary with the faces detected
        # data.update({"num_faces": len(rects), "faces": rects, "success": True})

        print "time", time.time() - start
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_color = img[y:y + h, x:x + w]
            cv2.imshow('image', roi_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def video():
    FACE_DETECTOR_PATH = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

    video_capture = cv2.VideoCapture(0)

    i = 120
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        start = time.time()
        
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img = frame[y:y + h, x:x + w]

        # if time.time()-start >= 20:  
        print "Taken image", i
        cv2.imwrite('newtest/%s.jpg' % i , img)
        
        if i%20==0 and i!=0:
            # i=0
            # i+=1
            time.sleep(10)
            start = time.time()
            # break
        i+=1
        # time.sleep(20)
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # video()
    # preprocess()
    one_hot_names()
