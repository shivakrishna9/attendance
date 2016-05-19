# from recognise.resnet import *
from recognise import  class_net
from extrafiles.utility import *
import sys
sys.setrecursionlimit(100000)

plist = pre_process()
# x, y = get_input.from_file()
# x_test, y_test = get_input.test_file()

# resnet = resnet.ResNet()
# # resnet.get_data(x, y, x_test, y_test)
# resnet.resnet101()
# resnet.compile_net()
# # resnet.normalise_data()
# resnet.train_net(nb_epoch=1, batch_size=4)
# resnet.epsw(batch=4)

class_net.VGGNet(plist)
