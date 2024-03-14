import numpy as np
import matplotlib.pyplot as plt

#a)
img = plt.imread("road.jpg")
img = img[:,:,0].copy()
print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img, cmap="gray")
plt.imshow(img, alpha=0.5,cmap="gray")
# plt.show()

#b)
height, width = img.shape
plt.figure()
plt.imshow(img[:, int(width*(1/4)):int(width*(1/2))], cmap="gray")
# plt.show()

#c)
rotated = np.ones((width,width))
for i in range(height):
    for j in range(width):
        rotated[j][width-i-1] = img[i][j]
rotated = rotated[:, width-height:]
plt.figure()
plt.imshow(rotated,cmap="gray")
# plt.show()

#d)
mirrored = np.fliplr(img)
plt.figure()
plt.imshow(mirrored,cmap="gray")
plt.show()
