# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 23:37:10 2020

@author: Nabeel-IT
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Nabeel-IT/Downloads/Database/Poly_dataSet.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(np.reshape(y,(10,1)))

# Fitting Linear Regression to the dataset
from sklearn.svm import SVR
lin_reg = SVR(kernel ='rbf')
lin_reg.fit(X, y)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(X,y)


# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Prdict by SV Regression
lin_reg.predict(np.reshape(6.5,(1,1)))

# Predicting a new result with SV Regression
predict = sc_y.inverse_transform(lin_reg.predict(sc_X.transform(np.array([[6.5]]))))

#Prdict by Polynominal Regression
#lin_reg.predict(sc_y.fit_transform(6.5))