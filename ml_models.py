#!usr/bin/python

# === load libraries ===

# key libraries
import numpy as np
import pandas as pd
import simplejson as json
import math
import random

# data prep
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import sklearn.model_selection as curves

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

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
import util_plot
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from projects.boston_housing import visuals as vs
#import visuals as vs
#import pylab as pl ### not needed...?

# data prep
import project_data                     # custom library for


# === test functions ===

def entropy(p1, p2):
    '''calculate entropy for population of classification within an attribute'''
    return (-p1 * math.log(p1,2) -p2 * math.log(p2,2))

def rscores(project):
    ''' compute r-scores for features and target '''
    result              = []
    max_length          = len(max(s.X.keys(), key=len)) + 1
    for k, feature in enumerate(project.X):
        feature_series  = project.X[feature].values.reshape(-1,1)
        reg = LinearRegression()
        reg.fit(feature_series, project.y)
        # coef            = reg.coef_[0] # with multiple regression model there are multiple coefficients
        # inter           = reg.intercept_
        score           = reg.score(feature_series, project.y)
        # r_score         = "r-square: {0:.3f}".format(score)
        result.append((feature, score))
    result.sort(key=lambda tup: tup[1], reverse=True)
    print("\tfeature".ljust(max_length) + "score".rjust(10))
    for feature, score in result:
        print("\t{0}".format(feature).ljust(max_length) + "{0:.1%}".format(score).rjust(10))
    return result


def computeLinearRegression(sleep,scores):
    #	First, compute the average amount of each list
    avg_sleep = np.average(sleep)
    avg_scores = np.average(scores)

    #	Then normalize the lists by subtracting the mean value from each entry
    normalized_sleep = [i-avg_sleep for i in sleep]
    normalized_scores = [i-avg_scores for i in scores]
#     normalized_sleep = ['{:.2f}'.format(i-avg_sleep) for i in sleep]
#     normalized_scores = ['{:.2f}'.format(i-avg_scores) for i in scores]
    print(normalized_sleep)
    print(normalized_scores)

    #	Compute the slope of the line by taking the sum over each student
    #	of the product of their normalized sleep times their normalized test score.
    #	Then divide this by the sum of squares of the normalized sleep times.
    slope = sum([x*y for x,y in zip(normalized_sleep,normalized_scores)]) / sum([np.square(y) for y in normalized_sleep])# = 0
    print(slope)
    #	Finally, We have a linear function of the form
    #	y - avg_y = slope * ( x - avg_x )
    #	Rewrite this function in the form
    #	y = m * x + b
    #	Then return the values m, b
    m = slope
    b = -slope*avg_sleep + avg_scores
    #y = m * x + b
    return m,b

def computePolynomialregression():
    # polynomial regression
    #y = p[0] * x**2 + p[1] * x + p[2]
    pass

#if __name__=="__main__":
def printLinearRegressionModel():
    m,b = compute_regression(sleep,scores)
    print("Your linear model is y={}*x+{}".format(m,b))

# === model object ===

class Model(object):
    ''' base model object '''
    test_size       = 0.20
    random_state    = 0
    n_splits        = 10
    params          = {'max_depth': list(range(1,11))}
    def __init__(self, project):
        self.project    = project_data.ProjectData(project)
        self.y          = self.project.target_data          # need to incorporate reg and lc data sets...
        self.X          = self.project.feature_data
        self.preprocessData()
        self.splitTrainTest()
#        self.fit_model()
    def preprocessData(self):
        ''' transpose objects to numerical data -> binary where appropriate '''
        # convert yes/no to 1/0
        print("\npreprocessing X & y, inputs and target values, replacing yes/no with 1/0")
        if self.y.dtype == object:          self.y.replace(to_replace=['yes', 'no'], value=[1, 0], inplace=True)
        print("\ty (target) values completed")
        for col, col_data in self.X.iteritems():
            if col_data.dtype == object:    self.X[col].replace(to_replace=['yes', 'no'], value=[1, 0], inplace=True)
        # use separate for loop to complete in place changes before processing 'get_dummies'
        print("\tX (input) values completed")
        for col, col_data in self.X.iteritems():
            if col_data.dtype == object:    self.X = self.X.join(pd.get_dummies(col_data, prefix = col))
        print("\tconverted categorical variable into dummy/indicator variables")
        # cast float64 to int64 for memory use and ease of reading (issue with get_dummies)
        for col in self.X.select_dtypes(['float64']):
            self.X[col] = self.X[col].astype('int64')
        print("\tconverted float to integer")
        # remove remaining object columns
        for col in self.X.select_dtypes(['O']):
            del self.X[col]
        print("\tremoved initial columns, now that they have been converted")
        self.all = pd.concat((self.X, self.y), axis=1)
        print("\tcreated 'all' dataframe, adding target as final column")
    def splitTrainTest(self):
        ''' use cross validation to split data into training and test datasets '''
        self.Xtr, self.Xt, self.Ytr, self.Yt = model_selection.train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
    def shuffleSplit(self):     # done inside of fit_model
        ''' use cross validation/shuffle to split data into training and test datasets '''
        self.cv_sets = ShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
    def getR2(self, y_true, y_predict):
        ''' calculate performance (aka coefficient of determination, goodness of fit) '''
        r2_score   = r2_score(y_true, y_predict)
        return(r2_score)
    def fit_model(self):
        """ Performs grid search over the 'max_depth' parameter for a
            decision tree regressor trained on the input data [X, y]. """
#     # Create cross-validation sets from the training data
        self.cv_sets = ShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
        self.regressor = DecisionTreeRegressor()
        # TODO: Create the grid search object
        grid = GridSearchCV(self.regressor, self.params, scoring = make_scorer(r2_score))
        # Fit the grid search object to the data to compute the optimal model
        grid = grid.fit(self.X, self.y)
        # Return the optimal model after fitting the data
        self.best_est = grid.best_estimator_

# === transform data ===

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
    return {    'model'             : model,
                'accuracy'          : accuracy,
                'confusion'         : confusion,
                'precision'         : precision,
                'recall'            : recall,
                'features_train'    : Xtr,
                'features_test'     : Xt,
                'labels_train'      : Ytr,
                'labels_test'       : Yt}

def plot_NB(n_iter=100):
    '''setup for Naive Bays learning curve'''
    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    estimator = GaussianNB()
    x,y,shape = setup_data()            # use data setup
    cv = cross_validation.ShuffleSplit(shape, n_iter=n_iter, test_size=0.2, random_state=0)
    print(cv)
    util_plot.plot_learning_curve(estimator, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
#    plt.show()

def plot_SVC(n_iter=10):
    '''setup for SVM RBF kernel, SVC is more expensive so do with lower number of CV iterations'''
    title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    x,y,shape = setup_data()            # use data setup
    cv = cross_validation.ShuffleSplit(shape, n_iter=n_iter, test_size=0.2, random_state=0)
    print(cv)
    estimator = SVC(gamma=0.001)
    util_plot.plot_learning_curve(estimator, title, x, y, (0.7, 1.01), cv=cv, n_jobs=4)
#    plt.show()

def plot_LR():
    '''setup for Linear Regression with KFold for cross validation'''
    title = "Learning Curves (Linear Regression)"
    size=1000
    X = np.reshape(np.random.normal(scale=2,size=size),(-1,1))
    y = np.array([[1 - 2*x[0] +x[0]**2] for x in X])
    cv = KFold(size,shuffle=True)
    print(cv)
    score = make_scorer(explained_variance_score)
    estimator = LinearRegression()
    util_plot.plot_learning_curve(estimator, title, X, y, (0.1, 1.01), cv=cv, scoring=score, n_jobs=4)
#    util_plot.plt.show()

def plot_DTReg():
    '''setup for Decision Tree Regressor'''
    title = "Learning Curves (Decision Tree Regressor)"
    size=1000
    cv = KFold(size,shuffle=True)           # Kfold (n, n_folds, shuffle, random_state)
    print(cv)
    score = make_scorer(explained_variance_score)
    X = np.round(np.reshape(np.random.normal(scale=5,size=2*size),(-1,2)),2)
    y = np.array([[np.sin(x[0]+np.sin(x[1]))] for x in X])
    estimator = DecisionTreeRegressor()
    util_plot.plot_learning_curve(estimator, title, X, y, (-0.1, 1.1), cv=cv, scoring=score, n_jobs=4)
#    plt.show()


## === functions from boston_housing ProjectData

def ModelLearning(X, y, tight={'rect':(0,0,0.75,1)}):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)
    # Create the figure window
    fig = util_plot.plt.figure(figsize=(10,7))
    # Create three different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        # Create a Decision tree regressor at max_depth = depth
        regressor = DecisionTreeRegressor(max_depth = depth)
        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(regressor, X, y, \
            cv = cv, train_sizes = train_sizes, scoring = 'r2')
        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)
        # Subplot the learning curve
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')
        # Labels
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
#     tight = {
#          'pad'      : 1,
#          'w_pad'    : 1,
#          'h_pad'    : 1,
#          'rect'     : (0,0,.75,0)
#     }
    fig.set_tight_layout(tight=tight)
    fig.show()
#    plt.set_tight_layout()
#    plt.show()

def ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """

    # Create 10 cross-validation sets for training and testing
#    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = curves.validation_curve(DecisionTreeRegressor(), X, y, \
        param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(max_depth, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(max_depth, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()


def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """
    # Store the predicted prices
    prices = []
    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size = 0.2, random_state = k)
        # Fit the data
        reg = fitter(X_train, y_train)
        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)
        # Result
        print("Trial {}: ${:,.2f}".format(k+1, pred))
    # Display price range
    print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))
