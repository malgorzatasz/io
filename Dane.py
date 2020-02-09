import pandas as pd

missing_values = ["NaN", "NA", "-", " "]
data=pd.read_csv('cardio_train.csv', sep=';', na_values =missing_values)

print(data.head())
print(data.dtypes)
print(data.describe())
print(data.isnull().any().any())
print(data.isnull().sum().sum())