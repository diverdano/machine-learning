import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
x = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

Xtr, Xt, Ytr, Yt = cross_validation.train_test_split(x, y, test_size=.25, random_state=0)

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

reg = DecisionTreeRegressor()
reg.fit(Xtr, Ytr)
DTR_mae = mae(reg.predict(Xt),Yt)
print "Decision Tree mean absolute error: {:.2f}".format(DTR_mae)

reg = LinearRegression()
reg.fit(Xtr, Ytr)
LR_mae = mae(reg.predict(Xt),Yt)
print "Linear regression mean absolute error: {:.2f}".format(LR_mae)

results = {
 "Linear Regression": DTR_mae,
 "Decision Tree": LR_mae