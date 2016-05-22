# This script captures the IP Camera feed and returns frames
import numpy as np
import cv2
import time
# import scikit.io


class Camera(object):

    def __init__(self, ip='192.168.1.64', user='admin', password='admin12345', request='/Output.h264'):
        self.ip = ip
        self.user = user
        self.password = password
        self.request = request
        self.req = 'rtsp://' + self.user + ':' + \
            self.password + '@' + self.ip + self.request
        self.cam = cv2.VideoCapture('small.mp4')

    def read_cam(self):
        ret, frame = self.cam.read()
        return frame

    def surveillance(self):
        # FACE_DETECTOR_PATH = "../extras/haarcascade_frontalface_default.xml"
        # faceCascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

        start = time.time()
        i=0
        while 1:
            ret, frame = self.cam.read()
            # print frame

            cv2.imshow('IP Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            i+=1

        # # When everything is done, release the capture
        self.cam.release()
        cv2.destroyAllWindows()
        print time.time() - start


if __name__ == '__main__':
    obj = Camera()
    # obj.__init__()
    obj.surveillance()
