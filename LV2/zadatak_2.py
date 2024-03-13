import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',')
data = np.delete(data, 0, 0)
#a)
print(f"Amount of data: {len(data)}")

# #b)
# plt.scatter(data[:,1], data[:,2], marker='.', color='red')
# plt.xlabel("Visina")
# plt.ylabel("Tezina")
# plt.show()

# #c)
# plt.scatter(data[::50, 1], data[::50, 2], marker='.', color='red')
# plt.xlabel("Visina")
# plt.ylabel("Tezina")
# plt.show()

#d)
print(f"Tallest person: {max(data[:,1])}")
print(f"Shortest person: {min(data[:,1])}")
print(f"Average height: {np.mean(data[:,1])}")

#e)
# ind = (data[:,0] == 1)
# print(ind)
men = []
women = []
for i in range(len(data)):
    if data[i,0] == 1:
        men.append(data[i,1])
    else:
        women.append(data[i,1])
men = np.array(men)
women = np.array(women)
print(f"Average male height:{np.mean(men)}")
print(f"Average female height:{np.mean(women)}")