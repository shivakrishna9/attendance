# from recognise.resnet import *
from recognise.class_net import *
from recognise.get_input import *
from extrafiles.utility import *
import sys
sys.setrecursionlimit(100000)

people = pre_process()

rec = VGG()
rec.VGGNet(people)
rec.evaluate(people)
