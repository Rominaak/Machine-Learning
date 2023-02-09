##Romina Akhavan Salmasi
##Assig48_2

#PRODUCTS LABELING

from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([[120, 50, 1],[60, 20, 2],
[145, 65, 1],[130, 45, 3],
[50, 15, 2]])
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit= neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)
from matplotlib import pyplot as plt
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()
#y opt = 11 (?)
model = DBSCAN(eps=26, min_samples=2)
model.fit(X)
print(model.labels_)
