import numpy as np
from LSTMCell import LSTMCell
from Linear import Linear, RepeatVector
from Metrics import softmax_cross_entropy, softmax
from typing import Tuple


class RNNAutoencoder:

    def __init__(self, vac_size: int, hidden_sizes: Tuple[int, int], seq_size: int):

        """
        Class implements RNNAutoencoder.

        Architecture of RNNAutoencoder have 2 lstm layers in encoder and
        2 lstm layers with linear layer in decoder

        :param vac_size: int
        :param hidden_sizes: Tuple[int, int]
        :param seq_size: int
        """

        self.vac_size = vac_size
        self.hidden_size_1 = hidden_sizes[0]
        self.hidden_size_2 = hidden_sizes[1]
        self.seq_size = seq_size

        #Encode
        self.lstm1 = LSTMCell(vac_size=self.vac_size, hidden_size=self.hidden_size_1, return_seq=True)
        self.lstm2 = LSTMCell(vac_size=self.hidden_size_1, hidden_size=self.hidden_size_2, return_seq=False)

        self.repeat = RepeatVector(self.seq_size)

        #Decode
        self.lstm3 = LSTMCell(self.hidden_size_2, self.hidden_size_1, return_seq=True)
        self.lstm4 = LSTMCell(self.hidden_size_1, self.vac_size, return_seq=True)

        self.linear = Linear(self.vac_size, self.vac_size)

    def params(self):
        """
        returns parameters of all model

        :return: dict
        """
        return {'lstm1': self.lstm1.params(),
                'lstm2': self.lstm2.params(),
                'lstm3': self.lstm3.params(),
                'lstm4': self.lstm4.params(),
                'linear': self.linear.params()}

    def clear_gradients(self):
        """
        function which clears gradients
        :return:
        """
        self.lstm1.clear_gradients()
        self.lstm2.clear_gradients()
        self.lstm3.clear_gradients()
        self.lstm4.clear_gradients()
        self.linear.clear_gradients()

    def forward(self, X: np.ndarray):
        """
        forward pass through the model

        :param X: np.ndarray
        :return: predictions of model
        """
        self.clear_gradients()

        encode = self.lstm2.forward(self.lstm1.forward(X))
        bridge = self.repeat.forward(encode)
        decode = self.lstm4.forward(self.lstm3.forward(bridge))

        decode = decode.reshape(decode.shape[0], decode.shape[1])

        pred = self.linear.forward(decode)

        return pred

    def compute_loss_and_gradient(self, X: np.ndarray, y: np.ndarray):
        """
        function which implement forward pass and calculation of loss and its derivative

        :param X: not-sorted one-hot array (seq_size, vac_size, 1)
        :param y: sorted sequence (seq_size, )
        :return: loss and its derivative
        """
        pred = self.forward(X)
        loss, dpredication = softmax_cross_entropy(pred, y)
        return loss, dpredication

    def repeat_backward(self, x: np.ndarray):
        """
        function which repeat vector for backward pass

        :param x: np.ndarray size (vac_size, 1)
        :return: d_out :np.ndarray size(seq_size, vac_size, 1)
        """
        d_out = np.zeros((self.seq_size, *x.shape))
        d_out[-1] = x
        return d_out

    def backward(self, d_out: np.ndarray):
        """
        backward pass through model

        :param d_out: derivative of loss
        :return:
        """
        d_l = self.linear.backward(d_out)
        d_l = d_l.reshape(*d_l.shape, 1)

        d_l = self.lstm3.backward(self.lstm4.backward(d_l))

        bridge = self.repeat_backward(self.repeat.backward(d_l))

        d_x = self.lstm1.backward(self.lstm2.backward(bridge))

        return d_x

    def predict(self, X: np.ndarray):
        """
        predict answer of the model
        :param X:
        :return:
        """
        pred = self.forward(X)
        probs = softmax(pred)
        return np.argmax(probs, axis=1)


class RNNAutoencoderOneLayer:
    """
    Class implements RNNAutoencoder.

    Architecture of RNNAutoencoder have 1 lstm layers in encoder and
    1 lstm layers with linear layer in decoder

    :param vac_size: int
    :param hidden_sizes: int
    :param seq_size: int
    """

    def __init__(self, vac_size: int, hidden_size: int, seq_size: int):

        self.vac_size = vac_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size

        #Encode
        self.lstm1 = LSTMCell(vac_size=self.vac_size, hidden_size=self.hidden_size, return_seq=False)

        self.repeat = RepeatVector(self.seq_size)

        #Decode
        self.lstm2 = LSTMCell(self.hidden_size, self.vac_size, return_seq=True)

        self.linear = Linear(self.vac_size, self.vac_size)

    def params(self):

        """
        returns parameters of all model

        :return: dict
        """

        return {'lstm1': self.lstm1.params(),
                'lstm2': self.lstm2.params(),
                'linear': self.linear.params()}

    def clear_gradients(self):

        """
        function which clears gradients
        :return:
        """

        self.lstm1.clear_gradients()
        self.lstm2.clear_gradients()
        self.linear.clear_gradients()

    def forward(self, X: np.ndarray):

        """
        forward pass through the model

        :param X: np.ndarray
        :return: predictions of model
        """

        self.clear_gradients()

        encode = self.lstm1.forward(X)
        bridge = self.repeat.forward(encode)
        decode = self.lstm2.forward(bridge)

        decode = decode.reshape(decode.shape[0], decode.shape[1])

        pred = self.linear.forward(decode)

        return pred

    def compute_loss_and_gradient(self, X: np.ndarray, y: np.ndarray):
        """
        function which implement forward pass and calculation of loss and its derivative

        :param X: not-sorted one-hot array (seq_size, vac_size, 1)
        :param y: sorted sequence (seq_size, )
        :return: loss and its derivative
        """
        pred = self.forward(X)
        loss, dpredication = softmax_cross_entropy(pred, y)
        return loss, dpredication

    def repeat_backward(self, x: np.ndarray):
        """
        function which repeat vector for backward pass

        :param x: np.ndarray size (vac_size, 1)
        :return: d_out :np.ndarray size(seq_size, vac_size, 1)
        """
        d_out = np.zeros((self.seq_size, *x.shape))
        d_out[-1] = x
        return d_out

    def backward(self, d_out: np.ndarray):
        """
        backward pass through model

        :param d_out: derivative of loss
        :return:
        """
        d_l = self.linear.backward(d_out)
        d_l = d_l.reshape(*d_l.shape, 1)

        d_l = self.lstm2.backward(d_l)

        bridge = self.repeat_backward(self.repeat.backward(d_l))

        d_x = self.lstm1.backward(bridge)

        return d_x

    def predict(self, X: np.ndarray):
        """
        predict answer of the model
        :param X:
        :return:
        """
        pred = self.forward(X)
        probs = softmax(pred)
        return np.argmax(probs, axis=1)






