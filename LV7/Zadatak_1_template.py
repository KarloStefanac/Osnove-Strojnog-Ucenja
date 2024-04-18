import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
# plt.show()

# flag 1=> 3 grupe 
# flag 2=> 3 grupe
# flag 3=> 4 grupe
# flag 4=> 2 grupe?
# flag 5=> 2 grupe?

#r Pimijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer
# obojite ovisno o njegovoj pripadnosti pojedinoj grupi. Nekoliko puta pokrenite programski
# kod. Mijenjate broj K. Što primjecujete?
km = KMeans(n_clusters=3)
km.fit(X)
labels = km.predict(X)
plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels)
# plt.show()

X2 = generate_data(500, 2)
km = KMeans(n_clusters=3)
km.fit(X2)
labels = km.predict(X2)
plt.figure()
plt.scatter(X2[:,0],X2[:,1], c=labels)

X3 = generate_data(500, 3)
km = KMeans(n_clusters=4)
km.fit(X3)
labels = km.predict(X3)
plt.figure()
plt.scatter(X3[:,0],X3[:,1], c=labels)

X4 = generate_data(500, 4)
km = KMeans(n_clusters=4)
km.fit(X4)
labels = km.predict(X4)
plt.figure()
plt.scatter(X4[:,0],X4[:,1], c=labels)

X5 = generate_data(500, 5)
km = KMeans(n_clusters=4)
km.fit(X5)
labels = km.predict(X5)
plt.figure()
plt.scatter(X5[:,0],X5[:,1], c=labels)

plt.show()
