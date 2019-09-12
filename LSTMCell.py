import numpy as np
from activations import sigmoid, tanh, dtanh
from Param import Param


class LSTMCell(object):
    """
    class which implements LSTMCell
    """
    def __init__(self, vac_size: int, hidden_size: int, return_seq: bool):
        self.hidden_size = hidden_size
        self.vac_size = vac_size
        self.return_seq = return_seq

        self.inner_size = self.vac_size + self.hidden_size

        self.W_forget = Param(np.random.uniform(low=-np.sqrt(1 / self.hidden_size), high=np.sqrt(1 / self.hidden_size),
                                                size=(self.hidden_size, self.inner_size)))
        self.b_forget = Param(np.random.uniform(low=-np.sqrt(1 / self.hidden_size), high=np.sqrt(1 / self.hidden_size),
                                                size=(self.hidden_size, 1)))

        self.W_input = Param(np.random.uniform(low=-np.sqrt(1 / self.hidden_size), high=np.sqrt(1 / self.hidden_size),
                                               size=(self.hidden_size, self.inner_size)))
        self.b_input = Param(np.random.uniform(low=-np.sqrt(1 / self.hidden_size), high=np.sqrt(1 / self.hidden_size),
                                               size=(self.hidden_size, 1)))

        self.W_cell_state = Param(
            np.random.uniform(low=-np.sqrt(1 / self.hidden_size), high=np.sqrt(1 / self.hidden_size),
                              size=(self.hidden_size, self.inner_size)))
        self.b_cell_state = Param(
            np.random.uniform(low=-np.sqrt(1 / self.hidden_size), high=np.sqrt(1 / self.hidden_size),
                              size=(self.hidden_size, 1)))

        self.W_output = Param(np.random.uniform(low=-np.sqrt(1 / self.hidden_size), high=np.sqrt(1 / self.hidden_size),
                                                size=(self.hidden_size, self.inner_size)))
        self.b_output = Param(np.random.uniform(low=-np.sqrt(1 / self.hidden_size), high=np.sqrt(1 / self.hidden_size),
                                                size=(self.hidden_size, 1)))

        self.cache = {}

        self.seq_size = 0

    def params(self):
        """
        parameters of LSTM layer
        :return:
        """
        return {'W_forget': self.W_forget, 'W_input': self.W_input,
                'W_cell_state': self.W_cell_state, 'W_output': self.W_output,
                'b_forget': self.b_forget, 'b_input': self.b_input,
                'b_cell_state': self.b_cell_state, 'b_output': self.b_output}

    def forget_gate(self, x: np.ndarray):
        """
        froget gate of the LSTM layer
        :param x:
        :return:
        """
        return sigmoid(np.dot(self.W_forget.value, x) + self.b_forget.value)

    def input_gate(self, x: np.ndarray):
        """
        input gate of the LSTM layer
        :param x:
        :return:
        """
        return sigmoid(np.dot(self.W_input.value, x) + self.b_input.value)

    def cell_state(self, x: np.ndarray):
        """
        cell state gate of the LSTM layer
        :param x:
        :return:
        """
        return tanh(np.dot(self.W_cell_state.value, x) + self.b_cell_state.value)

    def output_gate(self, x: np.ndarray):
        """
        output gate of the LSTM layer
        :param x:
        :return:
        """
        return sigmoid(np.dot(self.W_output.value, x) + self.b_output.value)

    def forward(self, seq: np.ndarray):
        """
        forward pass

        :param seq:
        :return:
        """

        h = Param(np.zeros((self.hidden_size, 1)))
        C = Param(np.zeros((self.hidden_size, 1)))
        ft = Param(np.zeros((self.hidden_size, 1)))
        it = Param(np.zeros((self.hidden_size, 1)))
        C_hat = Param(np.zeros((self.hidden_size, 1)))
        out = Param(np.zeros((self.hidden_size, 1)))

        self.seq_size = seq.shape
        self.cache = {}
        self.cache[-1] = (Param(h.value), Param(C.value))

        output = np.empty((self.seq_size[0], self.hidden_size, 1))

        for idx, x in enumerate(seq):

            x_oh = Param(np.row_stack((h.value, x)))

            ft.value = self.forget_gate(x=x_oh.value)
            it.value = self.input_gate(x=x_oh.value)
            C_hat.value = self.cell_state(x=x_oh.value)
            C.value = ft.value * C.value + it.value * C_hat.value
            out.value = self.output_gate(x=x_oh.value)
            h.value = out.value * tanh(C.value)

            output[idx] = h.value

            self.cache[idx] = (Param(x_oh.value), Param(ft.value),
                               Param(it.value), Param(C_hat.value),
                               Param(out.value), Param(h.value), Param(C.value))

        if self.return_seq:
            return output
        else:
            return output[-1]

    def backward(self, dh_out: np.ndarray):
        """
        backward pass

        :param dh_out:
        :return:
        """
        dh_from_next = np.zeros((self.hidden_size, 1))
        dC_from_next = np.zeros((self.hidden_size, 1))
        dx_out = np.empty((self.seq_size[0], self.vac_size, 1))
        for idx in range(len(self.cache.keys()) - 2, -1, -1):
            x_oh, ft, it, C_hat, out, h, C = self.cache[idx]
            C_prev = self.cache[idx - 1][-1]

            dh_out_cur = dh_out[idx]
            dh_out_cur += dh_from_next
            out.grad = dh_out_cur * tanh(C.value)
            C.grad = dC_from_next + dh_out_cur * out.value * dtanh(C.value)
            C_hat.grad = C.grad*it.value
            it.grad = C.grad*C_hat.value
            ft.grad = C.grad*C_prev.value
            ft.grad = ft.value * (1 - ft.value) * ft.grad
            out.grad = out.value * (1 - out.value) * out.grad
            C_hat.grad = (1 - C_hat.value * C_hat.value) * C_hat.grad
            it.grad = it.value * (1 - it.value) * it.grad

            x_oh.grad = np.dot(self.W_output.value.T, out.grad) + \
                        np.dot(self.W_cell_state.value.T, C_hat.grad) + \
                        np.dot(self.W_input.value.T, it.grad) + \
                        np.dot(self.W_forget.value.T, ft.grad)

            self.W_output.grad += np.dot(out.grad, x_oh.value.T)
            self.b_output.grad += out.grad

            self.W_cell_state.grad += np.dot(C_hat.grad, x_oh.value.T)
            self.b_cell_state.grad += C_hat.grad

            self.W_input.grad += np.dot(it.grad, x_oh.value.T)
            self.b_input.grad += it.grad

            self.W_forget.grad += np.dot(ft.grad, x_oh.value.T)
            self.b_forget.grad += ft.grad

            dh_from_next = x_oh.grad[:self.hidden_size, :]
            dx_out[idx] = x_oh.grad[self.hidden_size:, :]
            dC_from_next = ft.value * C.grad

        return dx_out

    def clear_gradients(self):
        """
        function which clears gradient of the layer
        :return:
        """

        self.W_output.grad = np.zeros_like(self.W_output.value)
        self.W_cell_state.grad = np.zeros_like(self.W_cell_state.value)
        self.W_input.grad = np.zeros_like(self.W_input.value)
        self.W_forget.grad = np.zeros_like(self.W_forget.value)

        self.b_output.grad = np.zeros_like(self.b_output.value)
        self.b_cell_state.grad = np.zeros_like(self.b_cell_state.value)
        self.b_input.grad = np.zeros_like(self.b_input.value)
        self.b_forget.grad = np.zeros_like(self.b_forget.value)
