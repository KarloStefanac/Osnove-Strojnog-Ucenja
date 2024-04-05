import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#a)
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=25, edgecolor='k', cmap=matplotlib.colormaps.get_cmap('coolwarm'))
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_test, s=25, cmap=matplotlib.colormaps.get_cmap('coolwarm'))
# plt.show()

#b)
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

#c)
print(LogRegression_model.coef_)
print(LogRegression_model.intercept_)

decision_border = -(LogRegression_model.coef_[0][1] / LogRegression_model.coef_[0][0])*X_train[:, 1] - (LogRegression_model.intercept_ / LogRegression_model.coef_[0][0])
plt.plot(X_train[:,1], decision_border, 'k-')
plt.show()

#d)
y_test_p = LogRegression_model.predict(X_test)
print("Tocnost: " + str(accuracy_score(y_test, y_test_p)))
cm = confusion_matrix(y_test, y_test_p)
print("Matrica zabune: " + str(cm))
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()
print(classification_report(y_test, y_test_p))

y_false = []
for i in range(len(y_test)):
    if y_test[i] != y_test_p[i]:
        y_false.append(1)
    else:
        y_false.append(0)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_false, s=25, cmap=matplotlib.colormaps.get_cmap('coolwarm'))
plt.show()