import numpy as np
# import tensorflow as tf
from keras import backend as K


def smooth(a):

    for x in a.shape[0]:
        if np.absolute(x) < 1:
            a[x] = K.square(x) / 2
        else:
            a[x] = np.absolute(x) - 0.5

    return a

def rcnn_reg(y_true, y_pred):

    k = np.array([smooth(j - y_true)])

    return k


def rcnn_cls(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.categorical_crossentropy(y_pred, y_true)


def rpn_cls(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.categorical_crossentropy(y_pred, y_true)


def rpn_reg(y_true, y_pred):

    k = np.array([smooth(j - y_true)])

    return k

from .utils.generic_utils import get_from_module


def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
