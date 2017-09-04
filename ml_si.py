#!usr/bin/python

# === load libraries ===

# key libraries
import numpy as np
import pandas as pd
import simplejson as json
from time import time

# data prep
from sklearn import model_selection     # for train_test_split

# models
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# ensemble: bagging, adaboost, random forest, gradient boosting
# KNeighbors
# stochastic gradient descent
from sklearn.svm import SVC
# logistic regression
# --not in sk learn? -- neural networks

'''
Gaussian Naive Bayes (GaussianNB)
Decision Trees
Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
K-Nearest Neighbors (KNeighbors)
Stochastic Gradient Descent (SGDC)
Support Vector Machines (SVM)
Logistic Regression
'''
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# insert here when confirmed
from sklearn.svm import SVC
# check these
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier # need gradient boosting
from sklearn.linear_model import LogisticRegression

# metrics
import sklearn.metrics
#from sklearn.metrics import f1_score, classification_score, confusion_matrix

# plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === data ===
def loadStudentData(file):
    return pd.read_csv(file)

# === test functions ===

# === plot ===
def plotCorr(data):
    '''plot correlation matrix for data (pandas DataFrame), exludes non-numeric attributes'''
    correlations    = data.corr()
    names           = list(correlations.columns)
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(names),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()


# === plot function ===
def correlation_matrix(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 130)
#    cmap = cm.get_cmap('jet', 30)
    cax = ax.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    # also check out matshow - plot a matrix as an image
    # also check out xcorr - plot a correlation x & y
    # minorticks_on()
#    ax.grid(True, markevery=1)
    ax.grid(True, markevery=1)
    plt.title('Project Data Feature Correlation')
    labels=list(df.columns)
#    labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    ax.set_xticklabels(labels, fontsize=6, minor=True, rotation='vertical')
    ax.set_yticklabels(labels, fontsize=6, minor=True, rotation='vertical')
    ax.set_xticks(range(0,len(labels)), minor=True)
    ax.set_yticks(range(0,len(labels)), minor=True)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    plt.show()

# === model object ===

class ProjectData(object):
    ''' get and setup data '''
    infile  = 'ml_projects.json'                    # should drop target/features from json? lift from data with pd.columns[:-1] & [-1]
    outfile = 'ml_projects_backup.json'
    df_col  = 10
    def __init__(self, project='boston_housing'):
        try:
            self.loadProjects()
            if project in self.projects.keys():
                self.desc       = project # if exists project in self.projects ...
                self.file       = self.projects[self.desc]['file']
                self.target     = self.projects[self.desc]['target']        # make y or move this to data, or change reg & lc samples?
                self.features   = self.projects[self.desc]['features']      # make X or move this to data, or change reg & lc samples?
                self.loadData()
                self.prepData()
            else:
                print('"{}" project not found; list of projects:\n'.format(project))
                print("\t" + "\n\t".join(list(self.projects.keys())))
        except: # advanced use - except JSONDecodeError?
            print('having issue reading project file...')
    def loadProjects(self):
        ''' loads project meta data from file and makes backup '''
        with open(self.infile) as file:
            self.projects  = json.load(file)
        with open(self.outfile, 'w') as outfile:
            json.dump(self.projects, outfile, indent=4)
    def saveProjects(self):
        ''' saves project meta detail to file '''
        with open(self.infile, 'w') as outfile:
            json.dump(self.projects, outfile, indent=4)
    def loadData(self):
        '''load data set as pandas.DataFrame'''
        self.data           = pd.read_csv(self.file)
        pd.set_option('display.width', None)                    # show columns without wrapping
        pd.set_option('display.max_columns', None)              # show all columns without elipses (...)
        pd.set_option('display.max_rows', self.df_col)               # show default number of rows for summary
        print("file loaded: {}".format(self.file))
        print("{} dataset has {} data points with {} variables each\n".format(self.desc, *self.data.shape))
        print("DataFrame Description (numerical attribute statistics)")
        print(self.data.describe())
        print("DataFrame, head")
        print(self.data.head())
    def prepData(self):
        '''split out target and features based on known column names in project meta data'''
        self.target_data    = self.data[self.target]
        self.feature_data   = self.data.drop(self.target, axis = 1)

# === transform data ===

class StudentData(object):
    ''' base model object '''
    test_size       = 0.24 # odd proportion given 395 records
    random_state    = 0
    n_splits        = 10
    params          = {'max_depth': list(range(1,11))}
    def __init__(self, project):
        self.project    = ProjectData(project)
        self.y          = self.project.target_data          # need to incorporate reg and lc data sets...
        self.X          = self.project.feature_data
        self.getDataSummary()
        self.preprocessData()
        self.splitTrainTest()
#        self.fit_model()
    def getDataSummary(self):
        ''' lift data attributes from DataFrame '''
        self.n_students = len(self.y.index)
        self.n_features = len(self.X.columns)
        passed_dict = self.y.value_counts().to_dict()
#        passed_dict = self.y.groupby(['passed'])['passed'].count().to_dict()
        self.n_passed = passed_dict['yes']
        self.n_failed = passed_dict['no']
        self.grad_rate = float(self.n_passed) / (self.n_passed + self.n_failed)

        # Print the results
        print("\nStudent Data Summary")
        print("\t{}\tstudents".format(self.n_students))
        print("\t{}\tfeatures".format(self.n_features))
        print("\t{}\tstudents passed".format(self.n_passed))
        print("\t{}\tstudents failed".format(self.n_failed))
        print("\t{:.2%}\tgraduation rate".format(self.grad_rate))
        print("\ntarget attribute:\n\t'{}'".format(self.y.name))
        print("\nfeatures:")
        for index, item in enumerate(sorted(self.X.columns)): print("\t{}\t'{}'".format(index + 1, item))
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
        print("\nsplitting test and train data sets with {} test size and {} random state".format(self.test_size, self.random_state))
        self.Xtr, self.Xt, self.Ytr, self.Yt = model_selection.train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

# === model ===
def StudentModel(object):
    '''model wrapper for StudentData'''
    def __init__(self, project):
        pass

    # TODO: Initialize the three models
    clf_A = DecisionTreeClassifier()
    clf_B = GaussianNB()
    clf_C = SVC()

    # TODO: Set up the training set sizes
    # ? use model_selection.train_test_split??
#    self.Xtr, self.Xt, self.Ytr, self.Yt = model_selection.train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
    X_train_100 = None
    y_train_100 = None

    X_train_200 = None
    y_train_200 = None

    X_train_300 = None
    y_train_300 = None

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)
        print("\ttest & train data sets split")
        '''
        Gaussian Naive Bayes (GaussianNB)
        Decision Trees
        Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
        K-Nearest Neighbors (KNeighbors)
        Stochastic Gradient Descent (SGDC)
        Support Vector Machines (SVM)
        Logistic Regression
        '''
    def setGaussianNB(self):
        '''Methods
        fit(X, y[, sample_weight])	Fit Gaussian Naive Bayes according to X, y
        get_params([deep])	Get parameters for this estimator.
        partial_fit(X, y[, classes, sample_weight])	Incremental fit on a batch of samples.
        predict(X)	Perform classification on an array of test vectors X.
        predict_log_proba(X)	Return log-probability estimates for the test vector X.
        predict_proba(X)	Return probability estimates for the test vector X.
        score(X, y[, sample_weight])	Returns the mean accuracy on the given test data and labels.
        set_params(\*\*params)	Set the parameters of this estimator.
        '''
        self.clf_GNB = GaussianNB()
        self.clf_GNB.fit(self.Xtr, self.Ytr)                        # fit with training data
        self.clf_GNB.pred_y = self.clf_GNB.predict(self.Xt)         # predict target values
        self.clf_GNB.score  = self.clf_GNB.score(self.Xt, self.Yt)  # score vs test data
        self.clf_GNB.classification_report  = sklearn.metrics.classification_report(self.Yt, self.clf_GNB.pred_y)
        self.clf_GNB.confusion_matrix       = sklearn.metrics.confusion_matrix(self.Yt, self.clf_GNB.pred_y)
        print("mean accuracy given test data/labels is {:.1%}".format(self.clf_GNB.score))
        print(self.clf_GNB.classification_report)
        print(self.clf_GNB.confusion_matrix)
#        self.clf_GNB.sigmas = sorted(zip(self.X.columns,self.clf_GNB.sigma_[0], self.clf_GNB.sigma_[1]), key=lambda x: x[1], reverse=True)  # sigma is variance of each parameter, theta is mean
#        print("Gaussian - Naive Bayes sigmas for each input")
#        for item in self.clf_GNB.sigmas: print("\t{:.4}\t{:.4}\t{}".format(item[1], item[2], item[0]))
#        return("mean accuracy given test data/labels is {:.1%}".format(self.clf_GNB.score))
    def setDecisionTree(self):
        '''Methods
        apply(X[, check_input])	Returns the index of the leaf that each sample is predicted as.
        decision_path(X[, check_input])	Return the decision path in the tree
        fit(X, y[, sample_weight, check_input, ...])	Build a decision tree classifier from the training set (X, y).
        fit_transform(X[, y])	Fit to data, then transform it.
        get_params([deep])	Get parameters for this estimator.
        predict(X[, check_input])	Predict class or regression value for X.
        predict_log_proba(X)	Predict class log-probabilities of the input samples X.
        predict_proba(X[, check_input])	Predict class probabilities of the input samples X.
        score(X, y[, sample_weight])	Returns the mean accuracy on the given test data and labels.
        set_params(\*\*params)	Set the parameters of this estimator.
        transform(\*args, \*\*kwargs)	DEPRECATED: Support to use estimators as feature selectors will be removed in version 0.19.
        '''
        self.clf_DT = DecisionTreeClassifier()
        '''
        DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)[source]¶
        '''
        self.clf_DT.fit(self.Xtr, self.Ytr)
        self.clf_DT.score       = self.clf_DT.score(self.Xt, self.Yt)
        self.clf_DT.importances = sorted(zip(self.X.columns, self.clf_DT.feature_importances_), key=lambda x: x[1], reverse=True)
#        self.clf_DT.importances = zip(self.X.columns, self.clf_DT.feature_importances_)
#        sorted(test, key=lambda x: x[1], reverse=True)
        print("mean accuracy given test data/labels is {:.1%}".format(self.clf_DT.score))
        print("decisionTree importances for each input")
        for item in self.clf_DT.importances: print("\t{:.2}\t{}".format(item[1], item[0]))
    def setEnsembleMethods(self):
        pass
    def setKNN(self):
        pass
    def setGradientDecent(self):
        pass
    def setSVM(self):
        pass
    def setLogisticRegression(self):
        pass
        '''Methods
        decision_function(X)	Predict confidence scores for samples.
        densify()	Convert coefficient matrix to dense array format.
        fit(X, y[, sample_weight])	Fit the model according to the given training data.
        fit_transform(X[, y])	Fit to data, then transform it.
        get_params([deep])	Get parameters for this estimator.
        predict(X)	Predict class labels for samples in X.
        predict_log_proba(X)	Log of probability estimates.
        predict_proba(X)	Probability estimates.
        score(X, y[, sample_weight])	Returns the mean accuracy on the given test data and labels.
        set_params(\*\*params)	Set the parameters of this estimator.
        sparsify()	Convert coefficient matrix to sparse format.
        transform(\*args, \*\*kwargs)	DEPRECATED: Support to use estimators as feature selectors will be removed in version 0.19.
        '''
        self.clf_LR = LogisticRegression()
        '''
        (penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)[source]¶
        '''
        self.clf_LR.fit(self.Xtr, self.Ytr)
        self.clf_LR.score = self.clf_LR.score(self.Xt, self.Yt)   # score vs test data
        return("mean accuracy given test data/labels is {:.1%}".format(self.clf_LR.score))
