import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)

def kNeighbours(X_train_n, y_train, X_test_n, y_test,k):
    KNN_model = KNeighborsClassifier(n_neighbors=k)
    KNN_model.fit(X_train_n, y_train)
    y_train_p_KNN = KNN_model.predict(X_train_n)
    y_test_p_KNN = KNN_model.predict(X_test_n)
    print("------------------------------------------------------")
    print("K: " + str(k) + " Neighbours: ")
    print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
    print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

    plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title("K=" + str(k) + " neighbours. Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
    plt.tight_layout()
    # plt.show()
    
def plotting(classifier,X_train_n, y_train, y_train_p, title):
    plot_decision_regions(X_train_n, y_train, classifier=SVC_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title(title + " Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
    plt.tight_layout()
    # plt.show()
    
# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("------------------------------------------------------")
print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Logisticka regresija. Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
# plt.show()

    # Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Izraˇcunajte toˇcnost
    # klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje. Usporedite
    # dobivene rezultate s rezultatima logistiˇcke regresije. Što primje´cujete vezano uz dobivenu
    # granicu odluke KNN modela?

# K=5
kNeighbours(X_train_n, y_train, X_test_n, y_test,5)
# Kako izgleda granica odluke kada je K = 1 i kada je K = 100?
#K=1
kNeighbours(X_train_n, y_train, X_test_n, y_test,1)
#K=1000
kNeighbours(X_train_n, y_train, X_test_n, y_test,100)
plt.show()

#6.5.2
# Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra K
# algoritma KNN za podatke iz Zadatka 1.
pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
param_grid = {'knn__n_neighbors': np.arange(1, 50)}
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print("Najbolji parametri: ", grid.best_params_)
print("Tocnost: ", grid.best_score_)
print("Tocnost na testnom skupu: ", grid.score(X_test, y_test))

#6.5.3
    # Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
    # te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
    # ovih hiperparametara utjeˇce na granicu odluke te pogrešku na skupu podataka za testiranje?
    # Mijenjajte tip kernela koji se koristi. Što primje´cujete?

# C=1,gamma=0.01
SVC_model = svm.SVC(kernel='rbf', C=1, random_state=42, gamma=0.01)
scores = cross_val_score(SVC_model, X_train, y_train, cv=5)
print("----------------------------------------------\nRBF:")
print(scores)
SVC_model.fit(X_train_n, y_train)
# Evaluacija modela SVC
y_train_p_SVC = LogReg_model.predict(X_train_n)
y_test_p_SVC = LogReg_model.predict(X_test_n)

plotting(SVC_model,X_train_n, y_train, y_train_p_SVC, "RBF, C=1, gamma=0.01")


# C=1,gamma=0.1
SVC_model = svm.SVC(kernel='rbf', C=1, random_state=42, gamma=0.1)
scores = cross_val_score(SVC_model, X_train, y_train, cv=5)
print("----------------------------------------------\nRBF:")
print(scores)
SVC_model.fit(X_train_n, y_train)
# Evaluacija modela SVC
y_train_p_SVC = LogReg_model.predict(X_train_n)
y_test_p_SVC = LogReg_model.predict(X_test_n)

plotting(SVC_model,X_train_n, y_train, y_train_p_SVC,"RBF, C=1, gamma=0.1")


# C=100,gamma=0.1
SVC_model = svm.SVC(kernel='rbf', C=100, random_state=42, gamma=0.1)
scores = cross_val_score(SVC_model, X_train, y_train, cv=5)
print("----------------------------------------------\npoly:")
print(scores)
SVC_model.fit(X_train_n, y_train)

# Evaluacija modela SVC
y_train_p_SVC = LogReg_model.predict(X_train_n)
y_test_p_SVC = LogReg_model.predict(X_test_n)

plotting(SVC_model,X_train_n, y_train, y_train_p_SVC,"RBF, C=100, gamma=0.1")

plt.show()


#6.5.4
    # Pomo´cu unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ
    # algoritma SVM za problem iz Zadatka 1.
pipe = Pipeline([('scaler', StandardScaler()), ('svm', svm.SVC(kernel='rbf'))])
param_grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': [0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print("Najbolji parametri: ", grid.best_params_)
print("Tocnost: ", grid.best_score_)
print("Tocnost na testnom skupu: ", grid.score(X_test, y_test))
