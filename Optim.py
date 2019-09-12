import numpy as np


class MomentumSGD:
    """
    class implements momentum sgd
    """
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.vel = 0

    def update(self, value: np.ndarray, dvalue: np.ndarray, lr: float):
        """
        function which updates value of parameter using momentum

        :param value:
        :param dvalue:
        :param lr:
        :return:
        """
        self.vel = self.momentum*self.vel - lr*dvalue
        value += self.vel