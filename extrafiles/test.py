import numpy as np
import cv2
import time
import glob
import os
import re
from collections import Counter
import random


def detect_haar(image):
    img = cv2.imread(image)

    FACE_DETECTOR_PATH = "../extras/haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    rects = detector.detectMultiScale(img, scaleFactor=1.03, minNeighbors=5,
                                      minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    faces = []
    images = []
    for (x, y, w, h) in rects:
        image = img[y:y + h, x:x + w]
        images.append(image)
        image = input_image(image)
        image = np.rollaxis(image, 2, start=0)
        # print image.shape
        faces.append(image)
    
    return faces

    # print faces
    # print np.array(faces).shape

    # return preprocess(np.array(faces), NB_CLASS=67), images

def rmfaces():
    for i in glob.glob('../extras/download/*/face/*.jpg'):
        print i
        os.remove(i)


def one_hot_names():
    lst = []
    with open("../traintest/lfwpeople.txt", 'r') as f:
        for line in f:
            lst += [line.split(',')[1]]
    print one_hot(",".join(lst), 5749, split=",")[:10]
    print lst[:10]

def pre_process():
    i = []
    for image in glob.glob("extras/train/*/*.jpg"):
        image = re.sub('\.\./', '', image)
        person = image.split('/')[2]
        img = image
        if 'not' in person:
            i.append((person, 'none', img))
        else:
            i.append(('face', person, img))

    up = [x for (a, x, y) in i if x != 'none']
    upx = up
    x = Counter(up)
    up = list(set(up))
    up = sorted(up)

    return up


def image_set():
    i = []
    for image in glob.glob("../extras/test1/*/*.jpg"):
        image = re.sub('\.\./', '', image)
        person = image.split('/')[2]
        img = image
        if 'not' in person:
            i.append((person, img))
        else:
            i.append((person, img))

    up = [x for (x, y) in i]
    upx = up
    x = Counter(up)
    up = list(set(up))
    up = sorted(up)
    # i.append(m)
    random.shuffle(i)

    with open("../traintest/race_test1.txt", 'w') as f:
        for k in i:
            if k[1] != 'none':
                # pass
                print k[0] + '\t' + str(up.index(k[0])) + '\t' + k[1]
                f.write(k[0] + '\t' + str(up.index(k[0])) + '\t' + k[1] + '\n')
            # else:
            #     print k[0]+'\t'+k[1]+'\t'+'none'+'\t'+k[2]
            #     f.write(k[0]+'\t'+k[1]+'\t'+'none'+'\t'+k[2]+'\n')

        # for k in m:
        #     if k[1] != 'none':
        #         # pass
        #         print k[1] + '\t' + str(up.index(k[1])) + '\t' + k[2]
        #         f.write(k[1] + '\t' + str(up.index(k[1])) + '\t' + k[2] + '\n')
            
    print up
    print len(upx)
    print x


def encode():
    lst = []
    l1 = []
    with open("../traintest/classtrain.txt", 'r') as f:
        for i in f:
            if i.split(',')[1].split('\n')[0] not in l1:
                l1.append(i.split(',')[1].split('\n')[0])
            lst.append([i.split(',')[1].split('\n')[0], i.split(',')[0]])

    print l1

    with open("train.txt", 'w') as f:
        for i in lst:
            print str(i[1]) + "," + str(l1.index(i[0]))
            f.write(i[1] + "," + str(l1.index(i[0])) + '\n')


def image():
    
    for image in glob.glob("../extras/2nd_sem/*/*.jpg"):
        start = time.time()
        r_path = '/'.join(image.split('/')[:-1])
        # im_race = image.split('/')[-1].split('\.')[0].split('_')[0]
        im_name = image.split('/')[-1].split('\.')[0]

        img = cv2.imread(image)
        # res = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)

        FACE_DETECTOR_PATH = "../extras/haarcascade_frontalface_default.xml"

        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        rects = detector.detectMultiScale(img,
                                          scaleFactor=1.03,
                                          minNeighbors=10,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        for i, (x, y, w, h) in enumerate(rects):
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_color = img[y:y + h, x:x + w]
            f_path = r_path + '/face/'
            if not os.path.exists(f_path):
                os.mkdir(f_path)
            print f_path  + im_name + '_' + str(i) + '.jpg'

            # cv2.imshow('image', roi_color)

            # if cv2.waitKey(0) & 0xFF == ord('y'):
            #     cv2.destroyAllWindows()
            cv2.imwrite(f_path + '/' + im_name + '_' +
                        str(i) + '.jpg', roi_color)

    # for image in glob.glob("../extras/test1/*.jpg"):
    #     start = time.time()
    #     r_path = '/'.join(image.split('/')[:-1])
    #     im_race = image.split('/')[-1].split('\.')[0].split('_')[0]
    #     im_name = image.split('/')[-1].split('\.')[0]

    #     img = cv2.imread(image)
    #     # res = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)

    #     FACE_DETECTOR_PATH = "../extras/haarcascade_frontalface_default.xml"

    #     detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    #     rects = detector.detectMultiScale(img,
    #                                       scaleFactor=1.03,
    #                                       minNeighbors=10,
    #                                       minSize=(30, 30),
    #                                       flags=cv2.CASCADE_SCALE_IMAGE)

    #     for i, (x, y, w, h) in enumerate(rects):
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #         roi_color = img[y:y + h, x:x + w]
    #         f_path = r_path + '/' + im_race
    #         if not os.path.exists(f_path):
    #             os.mkdir(f_path)
    #         print f_path + '/' + im_name + '_' + str(i) + '.jpg'

    #         # cv2.imshow('image', roi_color)

    #         # if cv2.waitKey(0) & 0xFF == ord('y'):
    #         #     cv2.destroyAllWindows()
    #         cv2.imwrite(f_path + '/' + im_name + '_' +
    #                     str(i) + '.jpg', roi_color)


            # elif cv2.waitKey(0) & 0xFF == ord('n'):
            #     cv2.destroyAllWindows()
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def video():
    FACE_DETECTOR_PATH = "../extras/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

    video_capture = cv2.VideoCapture(
        '../extras/vlc-record-2016-05-27-09h23m40s-rtsp___192.168.1.64_-.mp4')

    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        image = frame

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

        # cv2.imwrite('../extras/new/%s.jpg' % i , image[70:, 200:])
        # i+=1

        image = image[70:, 200:]

        # j = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (200, 70), (960,1280), (0, 255, 0), 2)
            img = image[y:y + h, x:x + w]
            cv2.imwrite('../extras/new/%s.jpg' % i , img)
            i+=1

        # if time.time()-start >= 20:
        # print "Taken image", i
        
        # Display the resulting frame
        cv2.imshow('Video', frame)

        # cv2.imwrite()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # image()
    # image_set()
    video()
