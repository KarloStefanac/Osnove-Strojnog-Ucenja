import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
for i in range(1,7):
    img = Image.imread(f"imgs/test_{i}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    # plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    # broj razlicitih boja u slici
    n_colors = len(np.unique(img_array, axis=0))
    print("Broj razlicitih boja u slici:",n_colors)
    km = KMeans(n_clusters=5)
    km = km.fit(img_array)
    print(km.cluster_centers_)
    labels = km.predict(img_array)

    # zamjena boja
    for i in range(w*h):
        img_array_aprox[i] = km.cluster_centers_[labels[i]]

    # prikaz nove slike
    img_array_aprox = np.reshape(img_array_aprox, (w,h,d))
    plt.figure()
    plt.title("5 razlitih boja")
    plt.imshow(img_array_aprox)
    plt.tight_layout()
    plt.show()
    
    
    k_values = range(1, 11)  
    inertia_values = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(img_array)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertia_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Broj grupa (K)')
    plt.ylabel('Inercija')
    plt.title('Ovisnost inercije o broju grupa (K)')
    plt.grid(True)
    plt.show()

    K = 5
    for i in range(K):
        binary_img = np.zeros((w*h, d))  
        binary_img[labels == i] = 1  
        binary_img = np.reshape(binary_img, (w, h, d))
        plt.figure()
        plt.imshow(binary_img, cmap='gray')  
        plt.title(f'Grupa {i+1}')
        plt.axis('off')
        plt.show()


