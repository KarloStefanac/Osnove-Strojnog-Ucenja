import numpy as np
import matplotlib.pyplot as plt
plt.figure()

plt.xlim(0, 4)
plt.ylim(0, 4)

x = np.linspace(1,3,2)
y = np.linspace(1,1,2)
plt.plot(x, y, 'o-', color='blue')
x = np.linspace(3,3,2)
y = np.linspace(1,2,2)
plt.plot(x, y, 'o-', color='blue')
x = np.linspace(2,3,2)
y = np.linspace(2,2,2)
plt.plot(x, y, 'o-', color='blue')
x = np.linspace(1,2,2)
y = np.linspace(1,2,2)
plt.plot(x, y, 'o-', color='blue')
plt.show()