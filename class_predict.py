# from recognise.resnet import *
from recognise import  class_net
from extrafiles.utility import *
import sys
sys.setrecursionlimit(100000)

plist = pre_process()
class_net.VGGNet(plist)
