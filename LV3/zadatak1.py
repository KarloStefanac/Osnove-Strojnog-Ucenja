import pandas as pd
data = pd.read_csv('data_C02_emission.csv')

#a)
print(len(data))
print(data.info())
print(f'Broj izostalih vrijednosti: {data.isnull().sum().sum()}')
print(f'Broj duplikata: {data.duplicated().sum()}')
data.drop_duplicates()
data.dropna(axis=1)
data.dropna(axis=0)
#Pretvaranje u category
data = data.apply(lambda x: x.astype('category') if x.dtype == 'object' else x)
print(data.info())
#b)
sorted_by_city = data.sort_values('Fuel Consumption City (L/100km)', ascending=True)
print(f"3 lowest fuel consumption cars in city: {sorted_by_city[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3)}")
print(f"3 highest fuel consumption cars in city:{sorted_by_city[['Make', 'Model', 'Fuel Consumption City (L/100km)']].tail(3)}")
#c)
engine_size = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print(f"Number of cars with an engine size between 2.5 and 3.5: {engine_size['Make'].count()}")
print(f"Average C02 emission of those cars: {engine_size['CO2 Emissions (g/km)'].mean():.2f} g/km")
#d)
audis = data[data['Make'] == 'Audi']
print(f"Number of Audi cars: {len(audis)}")
print(f"Average C02 emission of Audi cars with 4 cylinders: {audis[audis['Cylinders'] == 4]['CO2 Emissions (g/km)'].mean():.2f} g/km")
#e)
cylinders = data.groupby('Cylinders')
cylinders_mean = cylinders['CO2 Emissions (g/km)'].mean()
cylinders_count = cylinders['Make'].count()
print(pd.merge(cylinders_mean, cylinders_count, on='Cylinders').rename(columns={'CO2 Emissions (g/km)': 'Avg CO2 Emissions', 'Make': 'Number of cars'}))
#f)
diesel = data[data['Fuel Type'] == 'D']
petrol = data[data['Fuel Type'] == 'X']
print(f"Diesel city consumption average: {diesel['Fuel Consumption City (L/100km)'].mean():.2f} L/100km")
print(f"Petrol city consumption average: {diesel['Fuel Consumption City (L/100km)'].mean():.2f} L/100km")
print(f"Diesel city consumption median: {diesel['Fuel Consumption City (L/100km)'].median():.2f} L/100km")
print(f"Petrol city consumption median: {petrol['Fuel Consumption City (L/100km)'].median():.2f} L/100km")
#g)
four_cyl_diesel = diesel[diesel['Cylinders'] == 4]
four_cyl_diesel = four_cyl_diesel.sort_values('Fuel Consumption City (L/100km)', ascending=False)
print(f"Highest 4 cylinder diesel consumption car in city: {four_cyl_diesel[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(1)}")
#h)
manuals = data[data['Transmission'].str.startswith('M')]
print(f"Number of manual cars: {len(manuals)}")
#i)
print(f"Data correlation: {data.corr(numeric_only=True)}")