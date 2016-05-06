import numpy as np
import cv2
import time
import glob


def one_hot_names():
    lst = []
    with open("../traintest/lfwpeople.txt",'r') as f:
        for line in f:
            lst += [line.split(',')[1]]
    print one_hot(",".join(lst), 5749, split=",")[:10]
    print lst[:10]


def preprocess():
    
    with open("../traintest/classtrain.txt",'w') as f:
        for image in glob.glob("newtest/*/*.jpg"):
            print image.split('/')[2].split('.')[0]+","+image.split('/')[1]
            f.write(image.split('/')[2].split('.')[0]+","+image.split('/')[1]+'\n')


def encode():
    lst = []
    l1 = []
    with open("../traintest/classtrain.txt",'r') as f:
        for i in f:
            if i.split(',')[1].split('\n')[0] not in l1:
                l1.append(i.split(',')[1].split('\n')[0])
            lst.append([i.split(',')[1].split('\n')[0], i.split(',')[0]])

    print l1

    with open("train.txt",'w') as f:
        for i in lst:
            print str(i[1])+","+str(l1.index(i[0]))
            f.write(i[1]+","+str(l1.index(i[0]))+'\n')
            

def image_load():
    # Load an color image in grayscale

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # print img.shape
    for image in glob.glob("../extras/faceScrub/download/*/*.jpg"):
        start = time.time()
        img = cv2.imread(image)
        # res = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)

        FACE_DETECTOR_PATH = "../extras/haarcascade_frontalface_default.xml"

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
            cv2.imshow('image', img)
            if cv2.waitKey(0) & 0xFF == ord('y'):
                cv2.destroyAllWindows()
                with open("../traintest/detrain.txt",'a') as f:        
                    print image.split('/')[5]+","+image.split('/')[4]+','+str(x)+','+str(y)+','+str(w)+','+str(h)
                    f.write(image.split('/')[5]+","+image.split('/')[4]+','+str(x)+','+str(y)+','+str(w)+','+str(h)+'\n')

            elif cv2.waitKey(0) & 0xFF == ord('n'):
                cv2.destroyAllWindows()


if __name__ == '__main__':
    image_load()