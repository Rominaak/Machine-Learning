##Romina Akhavan Salmasi
##Assig48_1

#CARS

import numpy as np
x = np.array([[7, 3], [7, 4], [8, 3],[9, 3], [5, 2], [4, 3],[3,3]])
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors = 2)
neighbors_fit = neighbors.fit(x)
distances,indices = neighbors_fit.kneighbors(x)
from matplotlib import pyplot as plt
distances = np.sort(distances,axis = 0)
distances = distances[:,1]
plt.plot(distances)
plt.show()

#optimum y =1(eps opt = 1)

#ideal clustering

from sklearn.cluster import DBSCAN
model = DBSCAN(eps =1, min_samples=2)## optimum eps
model.fit(x)
print(model.labels_)
print(model.fit_predict([[8,4]]))
