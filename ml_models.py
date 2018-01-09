#!usr/bin/env python

# == load libraries ==

# key libraries
import numpy as np
import pandas as pd
import simplejson as json
import math
import random
from time import time

# runtime support
import os
import argparse                         # used when running as script
import warnings
import logging

# data prep
from sklearn import model_selection     # redundant?
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import sklearn.model_selection as curves

# models
''' for Udacity nano degree
Gaussian Naive Bayes (GaussianNB)
Decision Trees
Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
K-Nearest Neighbors (KNeighbors)
Stochastic Gradient Descent (SGDC)
Support Vector Machines (SVM)
Logistic Regression
'''
# TODO test these
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis # aka QDA
# insert here when confirmed
from sklearn.naive_bayes import GaussianNB                              # Naive Bayes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # Decision Tree
from sklearn.svm import SVC                                             # Support Vector Classifier / Support Vector Machines
from sklearn.linear_model import LinearRegression, LogisticRegression   # Logistic Regression
# check these
from sklearn.neighbors import KNeighborsClassifier                      # K-Nearest Neighbors
from sklearn.neural_network import MLPClassifier                        # Neural Network
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier # need gradient boosting
# --not in sk learn? -- neural networks

# metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.metrics import f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.feature_selection import RFE                               # feature ranking with recursive feature elimination

# custom
import util_plot        # plotting functions
import util_data     # data sourcing and pre-processing

## ===============================
## === command-line processing ===
## ===============================

parser = argparse.ArgumentParser()
parser.add_argument('--port',           default=8000,                           help="sets port number for web service")
parser.add_argument('--host',           default='localhost',                    help="sets host for web service")
parser.add_argument('--log_file',       default='log/web.log',                  help="path/file for logging")
parser.add_argument('--start',          dest='start',   action='store_true',    help="start the server")
parser.add_argument('--app',            default='simco_app',                    help="name of application")
parser.add_argument('--debug',          dest='debug',   action='store_true',    help="sets server debug, and level of logging")
parser.add_argument('--db',             default='sqlite:///simco.db',           help="designates the database to use")

args        = parser.parse_args()

def pargs():
    '''prints items in args object for ease of reading'''
    print('\n')
    for item in args._get_kwargs():
        k,v = item
        print('\t' + k + ': ' + str(v))

## ===================================
## === logging to file and console ===
## ===================================

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
#util.setLogLevel(args.debug, args.log_file)
#logger = logging.getLogger(args.app)
logger = logging.getLogger(__name__)
logger.info("set logger for app {0} - debug level as {1} using logfile: {2}".format(args.app, args.debug, args.log_file))


# == test functions ==

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

# == model object ==

class Model(object):
    ''' base model object '''
    test_size       = 0.20
    random_state    = 0
    n_splits        = 10
    params          = {'max_depth': list(range(1,11))}
    models = {
        "Nearest Neighbors"    : KNeighborsClassifier(3),
        "Linear SVM"           : SVC(kernel="linear", C=0.025),                                       # linear kernel
        "RBF SVM"              : SVC(gamma=2, C=1),                                                   # rbf kernel
        "Gaussian Process"     : GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        "Decision Tree"        : DecisionTreeClassifier(max_depth=5),
        "Random Forest"        : RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        "Neural Net"           : MLPClassifier(alpha=1),
        "AdaBoost"             : AdaBoostClassifier(),
        "Naive Bayes"          : GaussianNB(),
        "QDA"                  : QuadraticDiscriminantAnalysis(),
        "Logistic Regression"  : LogisticRegression(C=1e9)}                                           # added this one
    def __init__(self, project, split=False, score=False):
        self.project    = util_data.ProjectData(project)
        # self.preprocessData()
        if split:   self.splitTrainTest()
        if score:   self.fitNscore()
    def splitTrainTest(self):
        ''' use cross validation to split data into training and test datasets '''
        print("\n\tsplitting test and train data sets with {} test size and {} random state\n".format(self.test_size, self.random_state))
        self.Xtr, self.Xt, self.Ytr, self.Yt = model_selection.train_test_split(self.project.X, self.project.y, test_size=self.test_size, random_state=self.random_state)
        print("\tXtrain / Xtest = {} / {}".format(len(self.Xtr), len(self.Xt)))
        print("\tYtrain / Ytest = {} / {}".format(len(self.Ytr), len(self.Yt)))
    # def shuffleSplit(self):     # done inside of fit_model
    #     ''' use cross validation/shuffle to split data into training and test datasets '''
    #     self.cv_sets = ShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
    # def getR2(self, y_true, y_predict):
    #     ''' calculate performance (aka coefficient of determination, goodness of fit) '''
    #     r2_score   = r2_score(y_true, y_predict)
    #     return(r2_score)
    # def fit_model(self):
    #     """ Performs grid search over the 'max_depth' parameter for a
    #         decision tree regressor trained on the input data [X, y]. """
    #     # Create cross-validation sets from the training data
    #     self.cv_sets = ShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
    #     self.regressor = DecisionTreeRegressor()
    #     # TODO: Create the grid search object
    #     grid = GridSearchCV(self.regressor, self.params, scoring = make_scorer(r2_score))
    #     # Fit the grid search object to the data to compute the optimal model
    #     grid = grid.fit(self.X, self.y)
    #     # Return the optimal model after fitting the data
    #     self.best_est = grid.best_estimator_
    def fitNscore(self):
        '''quick fit and score of models'''
        print('\n\tfitting model, scoring and running predictions')
        print('\n\tnum\tscore\ttrain (s)\tpredict (s)\tdecision_function\tmodel\n')
        i = 1
        self.result = {}
        for name, model in self.models.items():
            startTr                     = time()
            # model.fit(self.Xtr, self.Ytr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(self.Xtr, self.Ytr)
            endTr                       = time()
            Tr_time                     = endTr - startTr
            startT                      = time()
            score = model.score(self.Xt, self.Yt)
            endT                        = time()
            T_time                      = endT - startT
            decision_function           = hasattr(model,"decision_function")    # test to find if decision_function exists, else predict_proba
            self.yt_pred                = model.predict(self.Xt)
            cm                          = confusion_matrix(self.Yt, self.yt_pred)
            # try:
            #     cm                      = confusion_matrix(self.Yt, self.yt_pred)
            # except Exception as e:
            #     print("cm-exception ({}): {}".format(name, e))
            #     cm                      = 'fail'
            # cr                      = classification_report(self.Yt, self.yt_pred)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cr                      = classification_report(self.Yt, self.yt_pred)
                # try:
                #     cr                      = classification_report(self.Yt, self.yt_pred)
                # except Exception as e:
                #     print("cr-exception ({}): {}".format(name, e))
                #     cr                      = 'fail'
            self.result[name]           = {
                                            "score"                 : score,
                                            "confusion_matrix"      : cm,
                                            "classification_report" : cr
                                        }
            print("\t{}\t{:.1%}\t{:.4f}\t\t{:.4f}\t\t{}\t\t{}".format(i, score, Tr_time, T_time, decision_function, name))
            i += 1
    def train_classifier(self):
        '''Fits a classifier to the training data and time the effort''' # Start the clock, train the classifier, then stop the clock
        start                           = time()
        self.clf.fit(self.Xtr, self.Ytr)
        end                             = time()
        print("\t{:.4f} seconds to train model".format(end - start))
    def predict_labels(self):
        '''Makes predictions using a fit classifier based on F1 score. Also provides accuracy''' # Start the clock, make predictions, then stop the clock
        pos_label                       = 1
        start                           = time()
        self.ytr_pred                   = self.clf.predict(self.Xtr)
        self.yt_pred                    = self.clf.predict(self.Xt)
        end                             = time()
        self.f1_score_Ytr               = f1_score(self.Ytr.values, self.ytr_pred, pos_label=pos_label)
        self.f1_score_Yt                = f1_score(self.Yt.values, self.yt_pred, pos_label=pos_label)
        self.accuracy_score             = accuracy_score(self.yt_pred, self.Yt)
#        self.classifier_score           = self.clf.score(self.Xt, self.Yt) # redundant
        self.classification_report      = classification_report(self.Yt, self.yt_pred)
        self.confusion_matrix           = confusion_matrix(self.Yt, self.yt_pred)
        # Print and return results
        print("\t{:.4f} seconds to make predictions".format(end - start))
        print("\t {:.1%} f1 score, training     (positive label is: {})".format(self.f1_score_Ytr, pos_label))
        print("\t {:.1%} f1 score, test         (positive label is: {})".format(self.f1_score_Yt, pos_label))
#        print("\t {:.1%} mean accuracy score    (subset accuracy)".format(self.classifier_score))
        print("\t {:.1%} mean accuracy score    (subset accuracy or jiccard similarity)".format(self.accuracy_score))
        print("\tclassification report:")
        print(self.classification_report)
        print("\tconfusion matrix:")
        print(self.confusion_matrix)

class MLModelExt(Model):
    '''extends Model, providing specific model tuning'''
    def setGaussianNB(self, verbose=False):
        ''''''
        self.clf = GaussianNB()
        self.train_classifier()
        self.predict_labels()
        if verbose:
            self.clf.sigmas = sorted(zip(self.Xt.columns,self.clf.sigma_[0], self.clf.sigma_[1]), key=lambda x: x[1], reverse=True)  # sigma is variance of each parameter, theta is mean
            print("Gaussian - Naive Bayes sigmas for each input")
            for item in self.clf.sigmas: print("\t{:.4}\t{:.4}\t{}".format(item[1], item[2], item[0]))
    def setDecisionTree(self, verbose=False):
        ''''''
        self.clf = DecisionTreeClassifier()
        '''DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)[source]'''
        self.train_classifier()
        self.predict_labels()
        self.clf.importances = sorted(zip(self.Xt.columns, self.clf.feature_importances_), key=lambda x: x[1], reverse=True)
        if verbose:
            print("\tdecisionTree importances for each input")
            for item in self.clf.importances: print("\t\t{:.2}\t{}".format(item[1], item[0]))
    def setAdaBoost(self):
        '''Ensemble Methods, ADA Boost Classifier'''
        self.clf = AdaBoostClassifier()
        self.train_classifier()
        self.predict_labels()
    def setVoting(self):
        '''Ensemble Methods, Voting Classifier'''
        self.clf = VotingClassifier()               #TODO add estimators
        self.train_classifier()
        self.predict_labels()
    def setRandomForest(self):
        '''Ensemble Methods, Random Forest'''
        self.clf = RandomForestClassifier()
        self.train_classifier()
        self.predict_labels()
    def setMLPC(self):
        '''Neural Network, MLPC Classifier'''
        self.clf = MLPClassifier()
        self.train_classifier()
        self.predict_labels()
    def setKNN(self):
        '''K-Nearest Neighbors'''
        self.clf = KNeighborsClassifier()
        self.train_classifier()
        self.predict_labels()
    def setSVM(self, kernel='linear', C=1, gamma=0.1, verbose=False, plot=False):
        self.kernel = kernel
        self.C      = C
        self.gamma  = gamma
        self.clf = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.train_classifier()
        self.predict_labels()
        if plot:
            '''callout to plot function'''
            prettyPicture(self.clf, self.Xt, self.Yt)
        if verbose:
            print("support vectors {}".format(self.clf.support_vectors_))
            # get indices of support vectors
            print("support {}".format(self.clf.support_))
            # get number of support vectors for each class
            print("n_support {}".format(self.clf.n_support_))
    def setLogisticRegression(self, C=1e9, verbose=False):
        '''
        decision_function(X)	Predict confidence scores for samples.
        predict_log_proba(X)	Log of probability estimates.
        predict_proba(X)	Probability estimates.
        '''
        self.clf        = LogisticRegression(C=C)
        '''(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)[source]'''
        self.train_classifier()
        self.predict_labels()
        if verbose:
            self.result          = pd.DataFrame(self.clf.coef_.transpose(),index=self.Xt.columns, columns=["coef"]) # create df with coefficients for each label
            self.result['abs']   = abs(self.result['coef'])
            pd.set_option('display.max_rows', 500)                      # show all features
            print('\tlabel coefficients')
            print(self.result.sort_values(by='abs', ascending=0))
    # def setGradientDecent(self):
    #     '''Stochastic Gradient Descent'''
    #     pass
    # def setBagging(self):   # find this library
    #     '''Ensemble Methods, Bagging'''
    #     self.clf = xyz()
    # def setGradientBoosting(self):   # find this library
    #     '''Ensemble Methods, Gradient Boosting'''
    #     self.clf = xyz()

class ModelSimple(object):
    '''for fine tuning models...?'''
#     kernel      = 'linear'
#     C           = 1
#     gamma       = 0.1
    accuracy    = None
    def __init__(self, project, kernel='linear', C=1, gamma=0.1):
        self.project    = util_data.ProjectData(project)
        self.kernel = kernel
        self.C      = C
        self.gamma  = gamma
        self.createClassifier()
        self.fitData()
        self.predict()
        self.getAccuracy()
    def __repr__(self):
        return str({'accuracy':format(self.accuracy,'0.3'), 'params':self.clf.get_params()})
    def createClassifier(self):
        self.clf = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        return self.clf.get_params()
    def fitData(self):
        self.clf.fit(self.project.X_train, self.project.y_train)
    def plotData(self):
        '''callout to plot function'''
        prettyPicture(self.clf, self.project.X_test, self.project.y_test)
    def predict(self):
        self.pred = self.clf.predict(self.project.X_test)
    def getAccuracy(self):
        self.accuracy = accuracy_score(self.pred, self.project.y_test)
        return self.__repr__()
    def get_support_vectors(self):
        clf.support_vectors_
        # get indices of support vectors
        clf.support_
        # get number of support vectors for each class
        clf.n_support_

class ModelNB(ModelSimple):
    '''inherit from model (ModelSimple), override create classifier with NB based'''
    def createClassifier(self):
        self.clf = GaussianNB()


# == transform data ==

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

# == specific plots ==

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


## == functions from boston_housing ProjectData ==

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
