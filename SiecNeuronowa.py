from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
from sklearn.preprocessing import normalize
from sklearn import metrics
import pandas as pd

data=pd.read_csv('cardio_train.csv', sep=';')

data = data.sample(frac=1.0)

X = data.drop(['id','cardio'], axis=1)
Y = data['cardio']

X_normalized=normalize(X,axis=0)

total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)

x_train=X_normalized[:train_length]
x_test=X_normalized[train_length:]
y_train=Y[:train_length]
y_test=Y[train_length:]

y_train=np_utils.to_categorical(y_train,num_classes=2)
y_test=np_utils.to_categorical(y_test,num_classes=2)

model=Sequential()
model.add(Dense(1000,input_dim=11,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=20,epochs=10,verbose=1)

y_pred=model.predict(x_test)

print("Sieć neuronowa")
print(metrics.accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print(metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))