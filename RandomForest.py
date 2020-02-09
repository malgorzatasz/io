import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data=pd.read_csv('cardio_train.csv', sep=';')

x = data.drop(['id','cardio'], axis=1)
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def randomforest(number):
    print("Random Forest estimators=%d" %(number))
    classifier = RandomForestClassifier(n_estimators=number)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))


randomforest(10)
randomforest(20)
randomforest(30)