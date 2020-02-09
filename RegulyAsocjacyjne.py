import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

data = pd.read_csv('cardio_train.csv', sep=';', usecols=['gender','cholesterol','gluc','smoke','alco','active','cardio'])

print(data.describe())

data = pd.get_dummies(data, columns = ['gender','cholesterol','gluc'])
print(data)

model = apriori(data, min_support=0.01, use_colnames=True)
rules = association_rules(model, metric="lift")
rules.sort_values('confidence', ascending = False, inplace = True)
print(rules.head(10))