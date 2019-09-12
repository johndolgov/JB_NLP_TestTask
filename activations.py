import numpy as np


def sigmoid(x: np.ndarray):
    """
    sigmoid function
    :param x:
    :return:
    """
    return 1/(1 + np.exp(-x))


def tanh(x: np.ndarray):
    """
    tanh function
    :param x:
    :return:
    """
    return np.tanh(x)


def dsigmoid(x: np.ndarray):
    """
    derivative of sigmoid
    :param x:
    :return:
    """
    return sigmoid(x)*(1-sigmoid(x))


def dtanh(x: np.ndarray):
    """
    derivative of dtanh
    :param x:
    :return:
    """
    return 1 - tanh(x)*tanh(x)
