import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50,50))
white = np.ones((50,50))
blackwhite = np.hstack((black,white))
whiteblack = np.hstack((white,black))
result = np.vstack((blackwhite,whiteblack))
plt.figure()
plt.imshow(result, cmap="gray")
plt.show()