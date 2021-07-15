from skimage.io import imread
from skimage.transform import resize

import matplotlib.pyplot as plt

from tensorflow.keras.applications.imagenet_utils import decode_predictions

from scivision_test_plugin import ImageNetModel

model = ImageNetModel()

x = imread('http://www.naturkrauter.net/image/net-250x250.png')
X = resize(x, (300, 224)) * 255    # cast back to 0-255 range

y = model.predict(X)

def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

plt.figure()
plt.imshow(x)
_, image_class, class_confidence = get_imagenet_label(y)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()