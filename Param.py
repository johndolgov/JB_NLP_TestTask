import numpy as np


class Param:
    """
    class Param which implements parameters with value and gradient
    """
    def __init__(self, value: np.ndarray):
        self.value = value
        self.grad = np.zeros_like(value)