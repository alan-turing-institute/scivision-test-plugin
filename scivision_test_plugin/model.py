import os

os.environ['TF_KERAS'] = '1'

import numpy as np
from skimage.filters import gaussian

if os.environ.get('TF_KERAS'):
    from classification_models.tfkeras import Classifiers


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
    ) -> np.ndarray:
        return gaussian(image, sigma=sigma, preserve_range=preserve_range)


class ImageNetModel:
    def __init__(self):
        self.pretrainedModel, self.preprocess_input = Classifiers.get('resnet18')

    def predict(self, image: np.ndarray) -> np.ndarray:

        image = self.preprocess_input(image)
        image = np.expand_dims(image, 0)

        y = model.predict(image)
        return y


if __name__ == "__main__":
    pass
