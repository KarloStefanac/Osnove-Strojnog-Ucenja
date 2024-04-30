import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import glob, os

model = keras.models.load_model('model.keras')

os.chdir("E:\\Faks\\Sesti semestar\\Osnove strojnog ucenja\\github\\Osnove-Strojnog-Ucenja\\LV8")
files = glob.glob("*.png")
print(files)
img_array = []
for file in files:
    img = mpimg.imread(file)
    print(img.shape)
    img = img[:, :, :1]
    img_array.append(img)

img_array = np.array(img_array)
img_array = img_array.reshape(-1, 28, 28, 1)
print(img_array.shape)
predictions = model.predict(img_array)
y_pred_classes = np.argmax(predictions, axis=1)
for i in range(len(img_array)):
    plt.imshow(img_array[i], cmap='gray')
    plt.title(f"True: {i}, Predicted: {y_pred_classes[i]}")
    plt.show()