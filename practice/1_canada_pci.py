import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import linear_model
from sklearn.svm import SVR
style.use("ggplot")

# Loading training data
df = pd.read_csv(r"F:\_deep_\lessons\codebasics\data\1_C_canada_income.csv")
# Renaming column for easy access
df.rename({'per capita income (US$)': 'pci'}, axis='columns', inplace=True)

# Linear Regression
reg = linear_model.LinearRegression()       # Linear Regression model object to hold data
reg.fit(df[['year']], df.pci)               # Feeding data to object

"""
Experimenting with a SVM - SVR that uses linear, polynomial
radial basis function etc.  (Learned from Siraj Rival)
We will try the radial basis function here 
"""
# Radial Basis Function
rbf = SVR(kernel='rbf', C=1e4, gamma=0.1)   # Configuring object for Radial Basis Function
rbf.fit(df[['year']], df.pci)               # Feeding data to RBF model object


# Plotting our Regression Models with Data-Points
plt.scatter(df[['year']], df.pci, color='red', marker='x', label='Data-Points')
plt.plot(df[['year']], reg.predict(df[['year']]), color='green', label='Linear Regression')
plt.plot(df[['year']], rbf.predict(df[['year']]), color='blue', label='Radial Basis Function')
plt.xlabel('Year')
plt.ylabel('Per Capita Income')
plt.title("Canada Per Capita Income by Year", fontsize=15)
plt.legend()
plt.show()

