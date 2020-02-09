from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

data=pd.read_csv('cardio_train.csv', sep=';')

x = data.drop(['id','cardio'], axis=1)
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def adaboost(number):
    print("AdaBoost estimators=%d" %(number))
    abc=AdaBoostClassifier(n_estimators=number)
    abc.fit(X_train, y_train)
    y_pred=abc.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

adaboost(10)
adaboost(20)
adaboost(30)