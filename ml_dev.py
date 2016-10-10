#!usr/bin/env python

# === libraries ===
# === base ===
import numpy as np
import pandas as pd

# === data prep ===
from sklearn import cross_validation
from sklearn.cross_validation import KFold

# === models ===


# === metrics ===


# === plot ===
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt

# === functions ===
def loadData(file = None, target = None):
    ''' load the dataset, default to titanic data set '''
    if file == None:
        X = pd.read_csv('~/machine-learning/projects/titanic_survival_exploration/titanic_data.csv')
        X = X._get_numeric_data()       # limit to numeric data
        y = X['Survived']           # default to titanic "Survived" attribute as target/label
        del X['Age'], X['Survived'] # remove 'Age' (missing data) and "Survived" target/label from input variables dataframe
        return {'features': X, 'labels': y}
    return pd.read_csv(file)

def consMake(makelist):
    '''consolidate list of automobile makers, e.g. VOLVO -> volvo'''
    for make in makelist:
        make = make.lower()

def plotData(data, title='title goes here', gridlines=True, legend=True, xlabel='x-label', ylabel='y-label'):
    '''plot data from dataframe'''
    plt.figure()
    if gridlines == True:   plt.grid()
    if legend == True:      plt.legend(loc='best')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if type(data) == pd.core.series.Series: plt.plot(data.index, data)  # need to improve the type test
    plt.show()

