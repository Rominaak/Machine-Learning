##Romina Akhavan Salmasi
##Assignment_48_3

#PRODUCTS LABELING

from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

X = pd.read_excel('Assig47.xlsx')
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit= neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)
from matplotlib import pyplot as plt
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()
#y opt = 29 (?)
model = DBSCAN(eps= 29, min_samples=2)
model.fit(X)
print(model.labels_)
