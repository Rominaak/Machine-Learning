##Romina Akhavan Salmasi
#Assig44_2

import numpy as np
import pandas as pd

data = pd.read_excel('Assig42_1.xlsx')
#print(data)
x = data[['a','b','c']]
y = data['d']

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x,y)
ypred = model.predict([[0,45,1]])
print(ypred)


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plot_tree(model)
plt.show()
