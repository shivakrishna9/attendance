import FaceRec
from FaceRec.net import *
from FaceRec.get_input import *

x,y = from_file()
VGGNet(x,y)
