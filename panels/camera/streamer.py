# This script captures the IP Camera feed and returns frames
import numpy as np
import cv2
import time

class Camera(object):

    def __init__(self, ip='192.168.1.64', user='admin', password='admin12345', request='/Output.h264'):
        self.ip = ip
        self.user = user
        self.password = password
        self.request = request
        self.req = 'rtsp://'+ self.user +':'+ self.password +'@'+ self.ip + self.request
        self.cam = cv2.VideoCapture(self.req)

    def read_cam(self):
        _, frame = self.cam.read()
        return frame

    def surveillance(self):
        FACE_DETECTOR_PATH = "../extras/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

        while (1):
            frame = self.read_cam()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=1,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img = image[y:y + h, x:x + w]

            cv2.imshow('IP Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # # When everything is done, release the capture
        self.cam.release()
        cv2.destroyAllWindows()
