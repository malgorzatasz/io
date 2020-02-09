from sklearn.cluster import KMeans
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix

data=pd.read_csv("PCA_cardio.csv")
x = data.iloc[:, 1:3].values

km= KMeans(n_clusters=2)
km.fit(x)
new_labels = km.labels_

clustering = DBSCAN(min_samples=10)
clustering.fit(x)
new_labels2=clustering.labels_

fig, axes = plt.subplots(1, 3, figsize=(12,6))
axes[0].scatter(data["principal component 1"], data["principal component 2"],edgecolor='k', s=150)
axes[1].scatter(data["principal component 1"], data["principal component 2"], c=new_labels, cmap='jet',edgecolor='k', s=150)
axes[2].scatter(data["principal component 1"], data["principal component 2"], c=new_labels2, cmap='jet',edgecolor='k', s=150)
axes[0].set_xlabel('principal component 1', fontsize=10)
axes[0].set_ylabel('principal component 2', fontsize=10)
axes[1].set_xlabel('principal component 1', fontsize=10)
axes[1].set_ylabel('principal component 2', fontsize=10)
axes[2].set_xlabel('principal component 1', fontsize=10)
axes[2].set_ylabel('principal component 2', fontsize=10)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=10)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=10)
axes[2].tick_params(direction='in', length=10, width=5, colors='k', labelsize=10)
axes[0].set_title('Przed grupowaniem', fontsize=10)
axes[1].set_title('Pogrupowane kMeans', fontsize=10)
axes[2].set_title('Pogrupowane DBSCAN', fontsize=10)
plt.show()