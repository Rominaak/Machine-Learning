##Romina Akhavan Salmasi
#Assig47

import pandas as pd
data = pd.read_excel('Assig47.xlsx')
states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Conneticut',
          'Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa']
#print(data)

#because the range of the data is large,
#the whole data will have a value of 0-1
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(data)
data1 = mms.transform(data)

from sklearn.cluster import KMeans

#elbow
se = []
for i in range(1,15):
    model = KMeans(n_clusters = i)
    model.fit(data1)
    se.append(model.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,15),se)
plt.xlabel('n_cluster')
plt.ylabel('elbow')
plt.show()
    
#n_cluster opt(elbow method) = 3


#KMeans
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
model.fit(data1)
a = model.labels_
print(a)
b = dict(zip(states,a))
print(b)
