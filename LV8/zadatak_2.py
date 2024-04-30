import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras import models
import seaborn as sns

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
model = keras.models.load_model('model.keras')

x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)

predictions = model.predict(x_test)
y_pred_classes = np.argmax(predictions, axis=1)

misclassified_indices = np.where(y_pred_classes != y_test)[0]

num_of_images = 9
plt.figure(figsize=(12, 8))
for i, idx in enumerate(misclassified_indices[:num_of_images]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Predicted: {y_pred_classes[idx]}")
    plt.axis('off')
plt.show()

