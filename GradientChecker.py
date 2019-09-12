import numpy as np
import LSTMCell
from Linear import Linear, RepeatVector
from RNNAutoencoder import RNNAutoencoder, RNNAutoencoderOneLayer
from Dataset import Dataset
from typing import Union


def check_gradient(f, x: np.ndarray, delta: float = 1e-6, tol: float = 1e-4):
    """
    compare analytical gradient and numeric gradient

    :param f: helper function which return loss and grad
    :param x: sequence
    :param delta: step in numeric gradient
    :param tol: tolerance
    :return:
    """
    _, analytic_grad = f(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        orig_x = x.copy()
        orig_x[ix] -= delta
        f_1, _ = f(orig_x)
        orig_x[ix] += 2*delta
        f_2, _ = f(orig_x)
        orig_x[ix] -= delta
        numeric_grad_at_ix = (f_2 - f_1)/(2*delta)
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print(f'Gradients are different at {ix}. Analytic: {analytic_grad_at_ix}. '
                  f'Numeric: {numeric_grad_at_ix}')
            return False
        it.iternext()
    print('Gradient check passed!')
    return True


def check_layer_gradient(layer: Union[Linear, LSTMCell.LSTMCell, RepeatVector],
                         x: np.ndarray, delta: float = 1e-5, tol: float = 1e-4):
    """
    check gradient of the layer

    :param layer:
    :param x:
    :param delta:
    :param tol:
    :return:
    """
    output = layer.forward(x)
    output_weight = np.random.randn(*output.shape)

    def helper_func(x: np.ndarray):
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        grad = layer.backward(d_out)
        return loss, grad

    return check_gradient(helper_func, x, delta, tol)


def check_layer_param_gradient(layer: Union[Linear, LSTMCell.LSTMCell, RepeatVector], x: np.ndarray,
                               param_name: str, delta: float = 1e-5, tol: float = 1e-4):
    """
    check parameters gradient in layer

    :param layer:
    :param x:
    :param param_name:
    :param delta:
    :param tol:
    :return:
    """

    param = layer.params()[param_name]
    initial_w = param.value

    output = layer.forward(x)
    output_weight = np.random.randn(*output.shape)

    def helper_func(w):
        param.value = w
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        layer.backward(d_out)
        grad = np.copy(param.grad)
        return loss, grad

    return check_gradient(helper_func, initial_w, delta, tol)


def check_model_gradient(model: Union[RNNAutoencoder, RNNAutoencoderOneLayer],
                         X: np.ndarray, y: np.ndarray, delta: float = 1e-5, tol: float = 1e-4):
    """
    check parameters gradients in the model
    :param model:
    :param X:
    :param y:
    :param delta:
    :param tol:
    :return:
    """
    for layer_name, params in model.params().items():
        for param_name, param in params.items():
            print(f'Checking gradient for {layer_name} parameter {param_name}')
            init_value = param.value

            def helper_func(value):
                param.value = value
                loss, dout = model.compute_loss_and_gradient(X, y)
                model.backward(dout)
                grad = param.grad
                return loss, grad

            if not check_gradient(helper_func, init_value, delta, tol):
                return False
    return True


def check_all_gradients(num_checks: int = 5):
    print('Checking Layers Only')
    print('Checking Linear Layer')
    for _ in range(num_checks):
        seq_size = np.random.randint(low=1, high=128)
        n_in = np.random.randint(low=1, high=128)
        n_out = np.random.randint(low=1, high=128)
        assert check_layer_gradient(Linear(n_in=n_in, n_out=n_out), np.random.randn(seq_size, n_in))
    print('Checking Linear Layer Paramter W')
    for _ in range(num_checks):
        seq_size = np.random.randint(low=1, high=128)
        n_in = np.random.randint(low=1, high=128)
        n_out = np.random.randint(low=1, high=128)
        assert check_layer_param_gradient(Linear(n_in=n_in, n_out=n_out), np.random.randn(seq_size, n_in), 'W')
    print('Checking Linear Layer Paramter b')
    for _ in range(num_checks):
        seq_size = np.random.randint(low=1, high=128)
        n_in = np.random.randint(low=1, high=128)
        n_out = np.random.randint(low=1, high=128)
        assert check_layer_param_gradient(Linear(n_in=n_in, n_out=n_out), np.random.randn(seq_size, n_in), 'b')
    print('Checking RepeatVector Layer')
    for _ in range(num_checks):
        seq_size = np.random.randint(low=1, high=128)
        n_in = np.random.randint(low=1, high=128)
        assert check_layer_gradient(RepeatVector(seq_size=seq_size), np.random.randn(n_in, 1))
    print('Checking LSTM Layer')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        hidden_size = np.random.randint(low=1, high=128)
        seq_size = np.random.randint(low=1, high=32)
        assert check_layer_gradient(LSTMCell.LSTMCell(vac_size=vac_size, hidden_size=hidden_size, return_seq=True),
                                    np.random.randn(seq_size, vac_size, 1))

    print('Checking LSTM Parameter W_forget')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        hidden_size = np.random.randint(low=1, high=32)
        seq_size = np.random.randint(low=1, high=32)
        assert check_layer_param_gradient(LSTMCell.LSTMCell(vac_size=vac_size, hidden_size=hidden_size, return_seq=True),
                                    np.random.randn(seq_size, vac_size, 1), 'W_forget')
    print('Checking LSTM Parameter W_input')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        hidden_size = np.random.randint(low=1, high=32)
        seq_size = np.random.randint(low=1, high=32)
        assert check_layer_param_gradient(LSTMCell.LSTMCell(vac_size=vac_size, hidden_size=hidden_size, return_seq=True),
                                    np.random.randn(seq_size, vac_size, 1), 'W_input')
    print('Checking LSTM Parameter W_cell_state')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        hidden_size = np.random.randint(low=1, high=32)
        seq_size = np.random.randint(low=1, high=32)
        assert check_layer_param_gradient(LSTMCell.LSTMCell(vac_size=vac_size, hidden_size=hidden_size, return_seq=True),
                                    np.random.randn(seq_size, vac_size, 1), 'W_cell_state')
    print('Checking LSTM Parameter W_output')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        hidden_size = np.random.randint(low=1, high=32)
        seq_size = np.random.randint(low=1, high=32)
        assert check_layer_param_gradient(LSTMCell.LSTMCell(vac_size=vac_size, hidden_size=hidden_size, return_seq=True),
                                    np.random.randn(seq_size, vac_size, 1), 'W_output')

    print('Checking LSTM Parameter b_forget')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        hidden_size = np.random.randint(low=1, high=32)
        seq_size = np.random.randint(low=1, high=32)
        assert check_layer_param_gradient(
            LSTMCell.LSTMCell(vac_size=vac_size, hidden_size=hidden_size, return_seq=True),
            np.random.randn(seq_size, vac_size, 1), 'b_forget')
    print('Checking LSTM Parameter b_input')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        hidden_size = np.random.randint(low=1, high=128)
        seq_size = np.random.randint(low=1, high=32)
        assert check_layer_param_gradient(
            LSTMCell.LSTMCell(vac_size=vac_size, hidden_size=hidden_size, return_seq=True),
            np.random.randn(seq_size, vac_size, 1), 'b_input')
    print('Checking LSTM Parameter b_cell_state')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        hidden_size = np.random.randint(low=1, high=32)
        seq_size = np.random.randint(low=1, high=32)
        assert check_layer_param_gradient(
            LSTMCell.LSTMCell(vac_size=vac_size, hidden_size=hidden_size, return_seq=True),
            np.random.randn(seq_size, vac_size, 1), 'b_cell_state')
    print('Checking LSTM Parameter b_output')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        hidden_size = np.random.randint(low=1, high=32)
        seq_size = np.random.randint(low=1, high=32)
        assert check_layer_param_gradient(
            LSTMCell.LSTMCell(vac_size=vac_size, hidden_size=hidden_size, return_seq=True),
            np.random.randn(seq_size, vac_size, 1), 'b_output')

    print('Checking All Two Layer Model Paramters')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        seq_size = np.random.randint(low=1, high=32)
        ds = Dataset(vac_size=vac_size, seq_size=seq_size)
        X, y = ds.generate_seq()
        assert check_model_gradient(model=RNNAutoencoder(vac_size=vac_size,
                                                         hidden_sizes=(12, 12), seq_size=seq_size), X=X, y=y)

    print('Checking All One Layer Model Paramters')
    for _ in range(num_checks):
        vac_size = np.random.randint(low=10, high=32)
        seq_size = np.random.randint(low=1, high=32)
        ds = Dataset(vac_size=vac_size, seq_size=seq_size)
        X, y = ds.generate_seq()
        assert check_model_gradient(model=RNNAutoencoderOneLayer(vac_size=vac_size, hidden_size=12, seq_size=seq_size), X=X, y=y)

    print('All Gradients Are Fine! Lets Train Model!')