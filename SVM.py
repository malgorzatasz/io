import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('cardio_train.csv', sep=';')

x = data.drop(['id','cardio'], axis=1)
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def svm(type):
      print("Kernel type = %s" %(type))

      svclassifier = SVC(kernel=type)
      svclassifier.fit(X_train, y_train)
      y_pred = svclassifier.predict(X_test)
      
      print(confusion_matrix(y_test,y_pred))
      print(accuracy_score(y_test,y_pred))

svm('rbf')
svm('sigmoid')
svm('poly')