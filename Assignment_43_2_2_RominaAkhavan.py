##Romina Akhavan Salmasi
#Assignment_43_2_2

import numpy as np
import pandas as pd

data = pd.read_excel('Assig42_1.xlsx')
#print(data)
x = data[['a','b','c']]
y = data['d']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier


    
model = KNeighborsClassifier(n_neighbors = 3)#from43_2_1
model.fit(xtrain,ytrain)
ypred = model.predict([[0,45,1]])


print(ypred)
