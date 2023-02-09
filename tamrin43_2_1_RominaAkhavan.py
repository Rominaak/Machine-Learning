#Romina Akhavan Salmasi
#Assig43_2_1

import numpy as np
import pandas as pd

data = pd.read_excel('Assig42_1.xlsx')
#print(data)
x = data[['a','b','c']]
y = data['d']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier
error = []

for i in range(1,20):
    
  model = KNeighborsClassifier(n_neighbors = i)
  model.fit(xtrain,ytrain)
  ypred = model.predict(xtest)
  error.append(np.mean(ypred!=ytest))

import matplotlib.pyplot as plt
plt.plot(range(1,20),error)
plt.xlabel('n_neighbors')
plt.ylabel('error')
plt.show()
print(ypred)

#n=3
