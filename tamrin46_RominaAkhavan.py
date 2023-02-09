##Romina Akhavan Salmasi
##Assig46

import numpy as np
x = np.array([[120, 50, 1],[60, 20, 2],
[145, 65, 1],[130, 45, 3],
[50, 15, 2]])

from sklearn.cluster import KMeans
#elbow
se = []
for i in range(1,5):
    model = KMeans(n_clusters = i, max_iter = 300)
    model.fit(x)
    se.append(model.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,5),se)
plt.xlabel('n_clusters')
plt.ylabel('elbow')
plt.show()


#n_cluster opt(elbow method) = 2



#silhouette
from sklearn.metrics import silhouette_score
sc=[]
for i in range(2,5):
    model = KMeans(n_clusters = i)
    model.fit(x)
    s = silhouette_score(x,model.labels_)
    sc.append(s)
plt.plot(range(2,5),sc)
plt.xlabel('n_clusters')
plt.ylabel('silhouette')
plt.show()


#n_cluster opt(silhouette method) = 2
