# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation


# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

Xtr, Xt, Ytr, Yt = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)


clf = DecisionTreeClassifier()
clf.fit(Xtr, Ytr)
dt_recall = recall(clf.predict(Xt),Yt)
dt_precision = precision(clf.predict(Xt),Yt)
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(dt_recall,dt_precision)

clf = GaussianNB()
clf.fit(Xtr, Ytr)
g_recall = recall(clf.predict(Xt),Yt)
g_precision = precision(clf.predict(Xt),Yt)
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(g_recall,g_precision)

results = {
  "Naive Bayes Recall": g_recall,
  "Naive Bayes Precision": g_precision,
  "Decision Tree Recall": dt_recall,
  "Decision Tree Precision": dt_precision
}