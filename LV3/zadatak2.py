import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

#a)
# plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20)
# plt.show()

#b)
# plt.figure()
color_map = {'X': 'red', 'Z': 'green', 'D': 'blue', 'E': 'yellow', 'N': 'black'}
colored_fuel = data['Fuel Type'].replace(color_map)
data.plot.scatter(x='Fuel Consumption City (L/100km)', y='CO2 Emissions (g/km)', c=colored_fuel, colormap='viridis')
handles = [plt.Line2D([0], [0], marker='o', color=color, label=fuel_type) for fuel_type, color in color_map.items()]
plt.legend(handles=handles)
# plt.show()

#c)
# plt.figure()
data.boxplot(column='Fuel Consumption Comb (L/100km)', by='Fuel Type')
# plt.show()

#d)
plt.figure()
fuel_type = data.groupby('Fuel Type')
fuel_type_count = fuel_type['Make'].count()

cylinders = data.groupby('Cylinders')
cylinders_co2 = cylinders['CO2 Emissions (g/km)'].mean()

plt.bar(fuel_type_count.keys(), fuel_type_count.values)
plt.bar(cylinders_co2.keys(), cylinders_co2.values)
plt.legend()
plt.show()
