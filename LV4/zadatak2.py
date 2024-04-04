from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import sklearn.linear_model as lm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv('data_C02_emission.csv')

ohe = OneHotEncoder ()
X_encoded = ohe.fit_transform(dataframe[['Fuel Type']]).toarray()

X = dataframe[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]
X = pd.concat([pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(['Fuel Type'])), X], axis=1)
y = dataframe['CO2 Emissions (g/km)']

linearModel = lm.LinearRegression()
linearModel.fit(X, y)
y_expect = linearModel.predict(X)

max_error = max(abs(y - y_expect))
print(max_error)

max_error_index = abs(y - y_expect).argmax()

data_point_X = X.iloc[max_error_index]
data_point_y = y.iloc[max_error_index]
data_point = dataframe.iloc[max_error_index]
print("X: ", data_point_X)
print("y: ", data_point_y)
print(data_point)
