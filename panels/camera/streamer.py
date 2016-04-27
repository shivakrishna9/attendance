import base64
import urllib2

import numpy as np
import cv2
import time



def printrec(recst):
    recs=recst.split('\r\n')
    for rec in recs:
        print rec


def video_cap():

    dest="DESCRIBE rtsp://admin:12345@192.168.1.74 RTSP/1.0\r\nCSeq: 2\r\nUser-Agent: python\r\nAccept: application/sdp\r\n\r\n"

    setu="SETUP rtsp://admin:12345@192.168.1.74/trackID=1 RTSP/1.0\r\nCSeq: 3\r\nUser-Agent: python\r\nTransport: RTP/AVP;unicast;client_port=60784-60785\r\n\r\n"

    play="PLAY rtsp://admin:12345@192.168.1.74/ RTSP/1.0\r\nCSeq: 5\r\nUser-Agent: python\r\nSession: SESID\r\nRange: npt=0.000-\r\n\r\n"

    # .. here SESID will be substituted with the session id that SETUP returns us ..

    ip="192.168.1.74"
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip,554))

    s.send(dest)
    recst=s.recv(4096)
    printrec(recst)




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
    cam = cv2.VideoCapture('rtsp://admin:admin12345@192.168.1.64:554/Output.h264')

    # i = 601
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video()