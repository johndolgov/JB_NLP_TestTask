import numpy as np
from Param import Param


class Linear:
    """
    class which implements Linear layer
    """
    def __init__(self, n_in: int, n_out: int):
        self.n_in = n_in
        self.n_out = n_out
        self.W = Param(np.random.randn(self.n_in, self.n_out) / np.sqrt(self.n_in/2))
        self.b = Param(np.random.randn(self.n_out))
        self.input_value = 0

    def params(self):
        """
        parameters of the linear layer
        :return:
        """
        return {'W': self.W, 'b': self.b}

    def forward(self, x: np.ndarray):
        """
        forward pass through the layer

        :param x:
        :return:
        """
        self.input_value = x
        return np.dot(x, self.W.value) + self.b.value

    def backward(self, grad_out: np.ndarray):
        """
        backward pass though the layer
        :param grad_out:
        :return:
        """
        grad_in = np.dot(grad_out, self.W.value.T)
        self.W.grad = np.dot(self.input_value.T, grad_out)
        self.b.grad = np.sum(grad_out, axis=0)
        return grad_in

    def clear_gradients(self):
        """
        function which clear gradients
        :return:
        """
        self.W.grad = np.zeros_like(self.W.value)
        self.b.grad = np.zeros_like(self.b.value)


class RepeatVector:
    """
    class which implements RepeatVector layer
    """
    def __init__(self, seq_size):
        self.seq_size = seq_size

    def params(self):
        """
        no parameters here
        :return:
        """
        return {}

    def forward(self, x: np.ndarray):
        """
        forward pass
        :param x:
        :return:
        """
        return np.tile(x.reshape(1, *x.shape), (self.seq_size, 1, 1))

    def backward(self, d_out: np.ndarray):
        """
        backward pass
        :param d_out:
        :return:
        """
        return np.sum(d_out, axis=0)

