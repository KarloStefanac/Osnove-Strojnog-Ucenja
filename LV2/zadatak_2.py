import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',')
data = np.delete(data, 0, 0)
#a)
print(f"Amount of data: {len(data)}")

#b)
plt.scatter(data[:,1], data[:,2], marker='x', color='green')
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Weigth-Heigth for every person")
plt.show()

#c)
plt.scatter(data[::50, 1], data[::50, 2], marker='.', color='red')
plt.xlabel("Visina")
plt.ylabel("Tezina")
plt.title("Weight-Height for every 50th person")
plt.show()

#d)
print(f"Tallest person: {max(data[:,1])}")
print(f"Shortest person: {min(data[:,1])}")
print(f"Average height: {np.mean(data[:,1])}")

#e)
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