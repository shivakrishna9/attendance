from recognise.resnet import *
from recognise.get_input import *

x, y = from_file()
x_test, y_test = test_file()

resnet = ResNet()
resnet.get_data(x, y, x_test, y_test)
resnet.resnet101()
resnet.compile_net()
resnet.normalise_data()
resnet.train_net(nb_epoch=3, batch=4)
resnet.epsw(batch=4)
