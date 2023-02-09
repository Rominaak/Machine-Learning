##RominaAkhavanSalmasi
##Assig42_1

import numpy as np
import pandas as pd

data = pd.read_excel('Assig42_1.xlsx')
#print(data)
x = data[['a','b','c']]
y = data['d']




#developing and training model

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)

#predicting
ypred = model.predict(xtest)
df = pd.DataFrame({'ytest':ytest,'ypred':ypred})
print(df)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))

print('predict = ', model.predict([[0,45,1]]))
