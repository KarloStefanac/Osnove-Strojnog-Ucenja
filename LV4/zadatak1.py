from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data_C02_emission.csv')
data = data.drop(0)
#a)
x_values = data[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)',
          'Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']]
y_values = data[['CO2 Emissions (g/km)']]
X_train, X_test, y_train, y_test = train_test_split(
    x_values, y_values, test_size=0.2, random_state=1)

#b)
plt.figure()
plt.scatter(X_train['Fuel Consumption Comb (mpg)'], y_train, color="blue",  s=10, alpha= 0.5)
plt.scatter(X_test['Fuel Consumption Comb (mpg)'], y_test, color="red",  s=10, alpha= 0.5)
# plt.show()

#c)
sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.fit_transform(X_test)
X_train_n = pd.DataFrame(X_train_n, columns=X_train.columns) # transformacija
X_test_n = pd.DataFrame(X_train_n, columns=X_train.columns) # transformacija s parametrima X_train skaliranja

plt.figure()
plt.hist(X_train['Fuel Consumption Comb (mpg)'], bins=20)
plt.xlabel('Fuel Consumption Comb (mpg)')
plt.title("Histogram prije normalizacije")

plt.figure()
plt.hist(X_train_n['Fuel Consumption Comb (mpg)'],bins=20)
plt.xlabel('Fuel Consumption Comb (mpg)')
plt.title("Histogram nakon normalizacije")
# plt.show()

#d)
print("Linear Regression for training:")
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)
print(linearModel.intercept_)

#e)
X_test_t = sc.transform(X_test)
y_test_p = linearModel.predict(X_test_t)
plt.figure()
plt.scatter(X_test['Fuel Consumption Comb (mpg)'], y_test, label='Real data',color="red",s=5)
plt.scatter(X_test['Fuel Consumption Comb (mpg)'], y_test_p, label='Predicted data',color="green",s=5)
plt.xlabel('Fuel Consumption Comb (mpg)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title("Linear Regression")
plt.legend(loc="upper right")
plt.show()

#f)
print("\nLinear Regression for testing:")
linearModel.fit(X_test_t,y_test)
print(linearModel.coef_)
print(linearModel.intercept_)

#g)
print("\nLinear Regression for half the test data:")
linearModel.fit(sc.transform(X_test[:int((len(X_test)-1)/2)]), y_test[:int((len(y_test)-1)/2)])
print(linearModel.coef_)
print(linearModel.intercept_)
print("\nLinear regression for 1/3 of the test data:")
linearModel.fit(sc.transform(X_test[:int((len(X_test)-1)/3)]), y_test[:int((len(y_test)-1)/3)])
print(linearModel.coef_)
print(linearModel.intercept_)
