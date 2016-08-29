#!usr/bin/python

import numpy as np
import pandas as pd

# data prep
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.datasets import load_linnerud      # linear regression data set

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score, make_scorer

# plot
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt




def loadData(file = '~/dev/machine-learning/projects/titanic_survival_exploration/titanic_data.csv'):
    '''load the dataset'''
    x = pd.read_csv(file)
    # Limit to numeric data
    x = x._get_numeric_data()
#     Separate the labels
    y = x['Survived']
#     Remove labels from the inputs, and age due to missing data
    del x['Age'], x['Survived']
    return {'features': x, 'labels': y}
#    return x

def loadDataReg():
    '''load regression data sample'''
    linnerud_data = load_linnerud()
    x = linnerud_data.data
    y = linnerud_data.target
    return {'data': x, 'target': y}



def splitTrainDataReg(x,y,test_size=0.25, random_state=0, model='Decision Tree'):
    '''split the data into training and testing sets then use classifier or regressor'''
    Xtr, Xt, Ytr, Yt = cross_validation.train_test_split(x, y, test_size=test_size, random_state=random_state)
    if model == 'Linear'      : reg = LinearRegression()
#    elif model == 'another'     : clf = OtherNB()
    else                        : reg = DecisionTreeRegressor()    # default is DT
    reg.fit(Xtr, Ytr)
    #     accuracy    = accuracy_score(reg.predict(Xt),Yt)
    #     confusion   = confusion_matrix(reg.predict(Xt),Yt)
    #     precision   = precision_score(reg.predict(Xt),Yt)
    #     recall      = recall_score(reg.predict(Xt),Yt)
    #     F1_score    = f1_score(reg.predict(Xt),Yt)
    #     F1_score_c  = 2 * (precision * recall) / (precision + recall)
    mae         = mean_absolute_error(reg.predict(Xt),Yt)
    mse         = mean_squared_error(reg.predict(Xt),Yt)
    r2          = r2_score(reg.predict(Xt),Yt)              # aka coefficient of determination
    exp_var     = explained_variance_score(reg.predict(Xt),Yt)
    print('\n' + model + 
#         '\n\tAccuracy:'.ljust(14)   + '{:.2f}'.format(accuracy) +
#         '\n\tF1 Score:'.ljust(14)   + '{:.2f}'.format(F1_score) +
#         '\n\tF1 Score_c:'.ljust(14) + '{:.2f}'.format(F1_score_c) +
        '\n\tMAE:'.ljust(14)        + '{:.2f}'.format(mae) +
        '\n\tMSE:'.ljust(14)        + '{:.2f}'.format(mse) +
        '\n\tR^2:'.ljust(14)        + '{:.2f}'.format(r2) +
        '\n\tMSE:'.ljust(14)        + '{:.2f}'.format(exp_var)
#         '\n\tPrecision:'.ljust(14)  + '{:.2f}'.format(precision) +
#         '\n\tRecall:'.ljust(14)     + '{:.2f}'.format(recall) +
#         '\n\tConfusion matrix: \n'
)
#     print(confusion)



def splitTrainData(x,y,test_size=0.25, random_state=0, model='Decision Tree'):
    '''split the data into training and testing sets then use classifier or regressor'''
    Xtr, Xt, Ytr, Yt = cross_validation.train_test_split(x, y, test_size=test_size, random_state=random_state)
    if model == 'Gaussian'      : clf = GaussianNB()
#    elif model == 'another'     : clf = OtherNB()
    else                        : clf = DecisionTreeClassifier()    # default is DT
    clf.fit(Xtr, Ytr)
    accuracy    = accuracy_score(clf.predict(Xt),Yt)
    confusion   = confusion_matrix(clf.predict(Xt),Yt)
    precision   = precision_score(clf.predict(Xt),Yt)
    recall      = recall_score(clf.predict(Xt),Yt)
    F1_score    = f1_score(clf.predict(Xt),Yt)
    F1_score_c  = 2 * (precision * recall) / (precision + recall)
    mae         = mean_absolute_error(clf.predict(Xt),Yt)
    mse         = mean_squared_error(clf.predict(Xt),Yt)
    print('\n' + model + 
        '\n\tAccuracy:'.ljust(14)   + '{:.2f}'.format(accuracy) +
        '\n\tF1 Score:'.ljust(14)   + '{:.2f}'.format(F1_score) +
        '\n\tF1 Score_c:'.ljust(14) + '{:.2f}'.format(F1_score_c) +
        '\n\tMAE:'.ljust(14)        + '{:.2f}'.format(mae) +
        '\n\tMSE:'.ljust(14)        + '{:.2f}'.format(mse) +
        '\n\tPrecision:'.ljust(14)  + '{:.2f}'.format(precision) +
        '\n\tRecall:'.ljust(14)     + '{:.2f}'.format(recall) +
        '\n\tConfusion matrix: \n')
    print(confusion)

# results = {
#   "Naive Bayes Recall": g_recall,
#   "Naive Bayes Precision": g_precision,
#   "Decision Tree Recall": dt_recall,
#   "Decision Tree Precision": dt_precision
# }

    return {    'model'             : model,
                'accuracy'          : accuracy,
                'confusion'         : confusion,
                'precision'         : precision,
                'recall'            : recall,
                'features_train'    : Xtr,
                'features_test'     : Xt,
                'labels_train'      : Ytr,
                'labels_test'       : Yt}

