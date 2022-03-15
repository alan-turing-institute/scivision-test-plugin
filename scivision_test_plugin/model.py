import numpy as np
from classification_models.tfkeras import Classifiers
from skimage.filters import gaussian
from skimage.transform import resize
from tensorflow.keras.applications.imagenet_utils import decode_predictions


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
    def __init__(self, model_name: str = "resnet18"):
        model, self.preprocess_input = Classifiers.get(model_name)

        # can build the model here, so we can reuse the prediction function
        self.pretrained_model = model(
            input_shape=(224, 224, 3), weights="imagenet", classes=1000
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        return "Test branch says Hello World!"


if __name__ == "__main__":
    pass
