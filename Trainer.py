import numpy as np
from RNNAutoencoder import RNNAutoencoder, RNNAutoencoderOneLayer
from Dataset import Dataset
from Optim import MomentumSGD
from copy import deepcopy
from typing import Union


class Trainer:
    """
    class implements training interface for model

    :param:model: RNNAutoencoder or RNNAutoencoderOneLayer
    :param:dataset
    :param:optimizer
    :param:num_epochs:int
    :param:train_size:int
    :param:lr: float
    :param:lr_decay: float
    """
    def __init__(self, model: Union[RNNAutoencoder, RNNAutoencoderOneLayer], dataset: Dataset,
                 optimizer: MomentumSGD, num_epochs: int,
                 train_size: int, lr: float = 1e-3, lr_decay: float = 1.0):

        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.num_epchs = num_epochs
        self.train_size = train_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizers = None

    def setup_optimizers(self):
        """
        setup optimizers for parameters
        :return:
        """
        layers = self.model.params()
        self.optimizers = {}
        for layer_name, params in layers.items():
            for param_name, param in params.items():
                self.optimizers[f'{layer_name}_{param_name}'] = deepcopy(self.optimizer)

    def fit(self):
        """
        model fit on the data which dataset generates

        :return: loss_history
        """
        if self.optimizers is None:
            self.setup_optimizers()

        loss_train_history = []
        for epoch in range(self.num_epchs):
            seq_loss = []
            for _ in range(self.train_size):
                X, y = self.dataset.generate_seq()
                loss, dpred = self.model.compute_loss_and_gradient(X, y)
                self.model.backward(dpred)

                for layer_name, params in self.model.params().items():
                    for param_name, param in params.items():
                        optimizer = self.optimizers[f'{layer_name}_{param_name}']
                        optimizer.update(param.value, param.grad, self.lr)
                seq_loss.append(loss)
            loss_train_history.append(np.mean(seq_loss))

            if len(loss_train_history) >= 2 and (np.isclose(loss_train_history[-1], loss_train_history[-2], 1e-2) or
                                                 loss_train_history[-1] > loss_train_history[-2]):

                print(f'LR reduction to {self.lr*self.lr_decay}')
                self.lr *= self.lr_decay

            print(f'Loss at epoch {epoch+1} is {np.mean(seq_loss)}')
        return loss_train_history
