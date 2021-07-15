import numpy as np
from skimage.filters import gaussian


class DummyModel:
    def __init__(self):
        pass

    def predict(self, image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
        return gaussian(image, sigma=sigma, preserve_range=True)


class AnotherModel:
    def __init__(self):
        pass

    def do_something(
        self,
        preserve_range: bool = True,
        sigma: float = 10.0,
        image: np.ndarray = np.empty((64, 64, 1)),
    ):
        return gaussian(image, sigma=sigma, preserve_range=preserve_range)


if __name__ == "__main__":
    pass
