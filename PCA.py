import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data=pd.read_csv('cardio_train.csv', sep=';')

x = data.drop(['id','cardio'], axis=1)
y = data['cardio']

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf.to_csv("PCA_cardio.csv")
finalDf = pd.concat([principalDf, data[['cardio']]], axis = 1)
finalDf.to_csv("PCA_cardio_with_target.csv")
