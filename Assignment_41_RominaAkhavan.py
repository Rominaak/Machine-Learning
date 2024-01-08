##RominaAkhavanSalmasi
#Assignment_41

import pandas as pd
data = pd.read_excel('Assig41.xlsx')
x = data[['a','b','c','d','e']]
y = data['f']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
ypred = model.predict([[70,1,2,17,2]])
print('ypred=',ypred,'million toman')
print(model.coef_)

