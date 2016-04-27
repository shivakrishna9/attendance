import base64
import urllib2

import numpy as np
import cv2
import time


class IpCamera(object):
    def __init__(self, url, user=None, password=None):
        self.url = url
        auth_encoded = base64.encodestring('{}:{}'.format(user, password))[:-1]
        self.req = urllib2.Request(self.url)
        self.req.add_header('Authorization', 'Basic {}'.format(auth_encoded))

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame

class Camera(object):

    def __init__(self, camera=0):
        self.cam = cv2.VideoCapture(camera)
        if not self.cam:
            raise Exception('Camera not accessible')
        self.shape = self.get_frame().shape

    def get_frame(self):
        _, frame = self.cam.read()
        return frame 

def video():
    cam = IpCamera('rtsp://192.168.1.64:554/Output.h264', user='admin', password='admin12345')

    # i = 601
    while True:
        # Capture frame-by-frame
        frame = cam.get_frame()

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video()