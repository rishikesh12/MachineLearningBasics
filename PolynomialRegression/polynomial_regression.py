# Bluffing Detector using Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting linear regression to the dataset

from sklearn.linear_model import LinearRegression
lin_reg =  LinearRegression()
lin_reg.fit(X,y)


# Fitting polynomial regression to the dataset

from sklearn.preprocessing import PolynomialFeatures

poly_reg =  PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

# visualizing linear regression results
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualizing polynomial regression results
X_grid = np.arange(min(X),max(X),0.1)   #stepwise incrementer for a proper plot
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Level vs Salary(Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()