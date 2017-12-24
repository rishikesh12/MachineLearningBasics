#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Data.csv')

#Matrix of features
X=dataset.iloc[:, :-1].values
# The first colon means all the rows, the -1 indicates all the columns except the last column

#Dependent variable vector
Y=dataset.iloc[:, 3].values

#Taking care of missing data

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer= imputer.fit(X[:, 1:3])
X[:,1:3]=imputer.transform(X[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()   #object of type LabelEncoder
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #returns the first column encoded

#encoding countries as 0,1,2 might indicate that one country is greater than the other which is not the case 
#hence we use onehotencoder to encode dummy variables
onehotencoder=OneHotEncoder(categorical_features=[0])
X= onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y= labelencoder_Y.fit_transform(Y)

#Splitting the dataset into training and test set

from sklearn.cross_validation import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)

#feature scaling age and salary are not on the same scale
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test) # no need for fit as sc_x was fit for the training set in the previous step



print(X)
print(Y)