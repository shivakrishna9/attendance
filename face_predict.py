# from recognise.resnet import *
from recognise import  net, resnet
from recognise.get_input import *
import time
import sys
sys.setrecursionlimit(100000)

<<<<<<< HEAD
# x, y, dets = imdb()
# x_test, y_test = test_file()
x, y = 1, 1
# resnet = ResNet()
# resnet.resnet101()
# resnet.compile()
# resnet.normalise_
# resnet.epsw()

while x and y:
    resnet = ResNet()
    resnet.resnet101()

    # x, y, dets = resnet.get_data()
    # print x, y, dets
    # print 'We are here once again'
=======

# x, y = get_input.from_file()
# x_test, y_test = get_input.test_file()

resnet = resnet.ResNet()
# resnet.get_data(x, y, x_test, y_test)
resnet.resnet101()
resnet.compile_net()
# resnet.normalise_data()
resnet.train_net(nb_epoch=1, batch_size=4)
resnet.epsw(batch=4)

# net.VGGNet(x,y,x_test,y_test)
# x, y = imdb()
# while x and y:
# 	time.sleep(5)
# 	print x,y
# 	x,y = imdb
>>>>>>> 511e48a4eb5789410fd446469eead4c97e7543d1
