import numpy as np
import matplotlib.pyplot as plt

#a)
img = plt.imread("road.jpg")
img = img[:,:,0].copy ()
print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img,cmap="gray")
plt.show()

#b)
height, width = img.shape
print(width*(3/4))
plt.figure()
plt.imshow(img[:, int(width*(3/4)):], cmap="gray")
plt.show()

#c)
black = np.zeros((50,50))
white = np.ones((50,50))
blackwhite = np.hstack((black,white))
whiteblack = np.hstack((white,black))
result = np.vstack((blackwhite,whiteblack))
plt.figure()
plt.imshow(result, cmap="gray")
plt.show()