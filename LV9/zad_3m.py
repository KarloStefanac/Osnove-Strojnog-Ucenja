#import bibilioteka
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import layers

#učitavanje dataseta
iris = datasets.load_iris()

#data = pd.DataFrame(iris.data, columns=iris.feature_names)
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(data)
print(iris['feature_names'])
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# Setosa - 0
# Versicolor - 1
# Virginica - 2


##################################################
#1. zadatak
##################################################

#a)
virginica = data[data['target'] == 2]
setosa = data[data['target'] == 0]
print(virginica.shape[0])
print(setosa.shape[0])

plt.scatter(virginica['petal length (cm)'], virginica['sepal length (cm)'], color="green", label="virginica")
plt.scatter(setosa['petal length (cm)'], setosa['sepal length (cm)'], color="gray", label="setosa")

plt.title("Odnos duljina latica i casica")
plt.xlabel("duljina latice (cm)")
plt.ylabel("duljina casice (cm)")
plt.legend()
plt.show()

# setosa su puno manji, i po duljini latice i casice, a virginica duzi

#b)
versicolor = data[data['target'] == 1]

x = ['Setosa', 'Versicolor', 'Virginica']
y = [setosa['sepal width (cm)'].max(), versicolor['sepal width (cm)'].max(), virginica['sepal width (cm)'].max()]


plt.bar(x, y)
plt.title('Najsire latice iy svake skupine')
plt.xlabel('skupina')
plt.ylabel('sirina (cm)')
plt.show()

# setosa ima najduzu laticu, ali su sve relativno blizu

#c)
avg = setosa['sepal width (cm)'].mean()
print(f"Setosa bigger than average: {setosa[setosa['sepal width (cm)'] > avg].shape[0]}")



##################################################
#2. zadatak
##################################################


#a) i b)
def predict(X, k):
    km = KMeans(n_clusters=k, init='random', n_init=5, random_state=0)
    km.fit(X)
    labels = km.predict(X)
    centers = km.cluster_centers_
    
    j = km.inertia_

    return [labels, centers, j]


X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = data['target']

labels, centers, j1 = predict(X, 2)
labels, centers, j2 = predict(X, 3)
labels, centers, j3 = predict(X, 4)
labels, centers, j4 = predict(X, 5)

j = [j1, j2, j3, j4]
plt.plot(range(2, 6), j)
plt.title('Elbow metoda')
plt.xlabel('vrijednost k')
plt.ylabel('vrijednost j')
plt.show()

#c)

km = KMeans(n_clusters = 3, init ='random', n_init =5, random_state =0)
km.fit(X)
labels = km.predict(X)
centers = km.cluster_centers_
print(centers)

#d)

cmap = []
for label in labels:
    if(label == 0):
        cmap.append('green')
    elif(label == 1):
        cmap.append('yellow')
    elif(label == 2):
        cmap.append('orange')


plt.scatter(data['petal length (cm)'], data['sepal length (cm)'], c=cmap)
plt.scatter(centers[:,2], centers[:,0], color = 'red', marker='x')
plt.xlabel('petal length (cm)')
plt.ylabel('sepal length (cm)')
plt.title('K srednjih vrijednosti (k=3)')

green_patch = mpatches.Patch(color='green', label='Setosa')
yellow_patch = mpatches.Patch(color='yellow', label='Versicolor')
orange_patch = mpatches.Patch(color='orange', label='Virginica')
plt.legend(handles=[green_patch, yellow_patch, orange_patch])

plt.show()


#e)

print("Tocnost: " + "{:0.3f}".format((accuracy_score(labels, data['target']))))


##################################################
#3. zadatak
##################################################

X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform((x_test))
y_train_s = keras.utils.to_categorical(y_train, 3)
y_test_s = keras.utils.to_categorical(y_test, 3)

#a)

model = keras.Sequential()
model.add(layers.Input(shape = (4, )))
model.add(layers.Dense(12, activation ="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(7, activation ="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation = "relu"))
model.add(layers.Dense(3, activation = "softmax"))
print(model.summary())

#b)
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy",])

#c)
model.fit(x_train, y_train_s, batch_size=7, epochs=450, validation_split=0.1)


#d)

model.save("model.keras")

del model

model = keras.models.load_model("model.keras")

#e)

score = model.evaluate( x_test, y_test_s, verbose =0)
print("Accuracy: ", score[1])
print("Loss: ", score[0])

#f)

predicted = model.predict(x_test)
predicted = np.argmax(predicted, axis=1)

print("Confusion matrix: \n", confusion_matrix(y_test, predicted))
