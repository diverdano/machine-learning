import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
x = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
Xtr, Xt, Ytr, Yt = cross_validation.train_test_split(x, y, test_size=.25, random_state=0)


reg = DecisionTreeRegressor()
reg.fit(Xtr, Ytr)
DT_mse = mse(reg.predict(Xt),Yt)
print("Decision Tree mean squared error: {:.2f}".format(DT_mse))

reg = LinearRegression()
reg.fit(Xtr, Ytr)
LR_mse = mse(reg.predict(Xt),Yt)
print("Linear regression mean squared error: {:.2f}".format(LR_mse))

results = {
 "Linear Regression": LR_mse,
 "Decision Tree": DT_mse
}