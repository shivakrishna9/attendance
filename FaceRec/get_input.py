import cv2
import pandas as pd

def detect(img):
    FACE_DETECTOR_PATH = "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    rects = detector.detectMultiScale(img, scaleFactor=1.4, minNeighbors=1,
                                      minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    # print "time", time.time() - start
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = img[y:y + h, x:x + w]
        cv2.imshow('image', roi_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return roi_color

def from_file(fname='train.txt'):
	read = pd.read_csv(fname,sep='\t',names=['filename','name'])

	images=[]
	image_classes=[]
	for data in read.itertuples():
		image = data[0]
		image_class = data[1]
		image_classes.append(image_class)
		image = input_image(image)
		images.append(image)

	return images, image_classes

def test_file(fname='test.txt'):
	read = pd.read_csv(fname,sep='\t',names=['filename','name'])
	return read

def input_image(img):

    res = detect(img)
    res = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)

    return res