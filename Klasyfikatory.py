from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

data=pd.read_csv('cardio_train.csv', sep=';')

x = data.drop(['id','cardio'], axis=1)
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Naive Bayes")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_bayes = gnb.predict(X_test)
accuracy_bayes=metrics.accuracy_score(y_test, y_pred_bayes)
print("Dokładność: %f" %(accuracy_bayes))
macierz_bayes=confusion_matrix(y_test, y_pred_bayes)
print(macierz_bayes)

print("Drzewo decyzyjne")
dtc = DecisionTreeClassifier()
decision_tree=dtc.fit(X_train, y_train)
y_pred_tree=dtc.predict(X_test)
accuracy_tree=metrics.accuracy_score(y_test, y_pred_tree)
print("Dokładność %f:" %(accuracy_tree))
macierz_tree=confusion_matrix(y_test, y_pred_tree)
print(macierz_tree)

def k_neighbours(number):
    print("k-NN, k=%d" %(number))
    knn = KNeighborsClassifier(n_neighbors=number, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred)
    print("Dokładność k=%d %f:" %(number,accuracy))
    macierz=confusion_matrix(y_test, y_pred)
    print(macierz)

k_neighbours(3)
k_neighbours(7)
k_neighbours(11)