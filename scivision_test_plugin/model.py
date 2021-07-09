import numpy as np
from skimage.filters import gaussian


class DummyModel:
    def __init__(self):
        pass

    def predict(self, data: np.ndarray, sigma: float = 3.0):
        return gaussian(data, sigma=sigma, preserve_range=True)


if __name__ == "__main__":
    pass
