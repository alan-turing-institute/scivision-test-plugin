import matplotlib.pyplot as plt
import numpy as np

from scivision_test_plugin import DummyModel

X = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
model = DummyModel()
y = model.predict(X)


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(X)
plt.subplot(1, 2, 2)
plt.imshow(y)
plt.show()
