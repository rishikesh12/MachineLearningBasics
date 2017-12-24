# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# SVR doesn't implement feature scaling we have to explicitly do it
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# fitting the SVR to the dataset

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')   # using a guassian kernal, there are multiple kernel options like linear,polynomial etc
regressor.fit(X,y)

# since we are using feature scaling we have to use inverse transform on the predicted value to get the predicted value in the oringinal format
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]))))

# Plotting the results
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Salary vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
