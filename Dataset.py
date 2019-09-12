import numpy as np


class Dataset:
    """
    Dataset class generate sequences
    """
    def __init__(self, vac_size: int, seq_size: int):

        self.vac_size = vac_size
        self.seq_size = seq_size

    def to_onehot(self, x: np.ndarray):
        """
        function which return one_hot presentation of sequences
        :param x:
        :return:
        """
        x_oh = np.zeros((self.seq_size, self.vac_size))
        x_oh[np.arange(self.seq_size), x] = 1
        return x_oh.reshape(*x_oh.shape, 1)

    def generate_seq(self):
        """
        generate sequence using vac_size and seq_size

        :return:
        """
        x = np.random.randint(low=0, high=self.vac_size, size=self.seq_size)
        y = np.sort(x)
        X = self.to_onehot(x)
        return X, y