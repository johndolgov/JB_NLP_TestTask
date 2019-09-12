import numpy as np


def cross_entropy_loss(probs: np.ndarray, target_index: np.ndarray):
    """
    cross_entropy function

    :param probs: np.ndarray
    :param target_index: np.ndarray
    :return: cross entropy loss
    """
    selection = probs[range(len(target_index)), target_index]
    return np.mean(-np.log(selection))


def softmax(pred: np.ndarray):
    """
    softmax function

    :param pred:
    :return: softmax function
    """
    prediction = np.copy(pred)
    maxs = np.max(prediction, axis=1)
    prediction -= maxs.reshape(*maxs.shape, 1)
    exps = np.exp(prediction)
    down = np.sum(exps, axis=1)
    probs = exps/down.reshape(*down.shape, 1)
    return probs


def softmax_cross_entropy(preds: np.ndarray, target_index: np.ndarray):
    """
    function which calculates probs and then cross entropy loss
    :param preds:
    :param target_index:
    :return: loss and dprediction
    """
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)

    subtr = np.zeros_like(probs)
    subtr[range(len(target_index)), target_index] = 1
    dprediction = (probs - subtr)/probs.shape[0]

    return loss, dprediction

