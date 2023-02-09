##RominaAkhavanSalmasi
##Assig43_3

import numpy as np
import pandas as pd

data = pd.read_excel('Assig42_2.xlsx')
#print(data)
x = data[['a','b']]
y = data['c']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier


    
model = KNeighborsClassifier(n_neighbors = 9)
model.fit(xtrain,ytrain)
ypred = model.predict([[40,2]])


print(ypred)
