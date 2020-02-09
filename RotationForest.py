import pandas as pd
from sklearn.model_selection import train_test_split
from rotation_forest import RotationForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv('cardio_train.csv', sep=';')

x = data.drop(['id','cardio'], axis=1)
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def rotationforest(number):
    print("Rotation Forest estimators=%d" %(number))
    rotation=RotationForestClassifier(n_estimators=number)
    rotation.fit(X_train, y_train)
    y_pred=rotation.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))


rotationforest(10)
rotationforest(20)
rotationforest(30)