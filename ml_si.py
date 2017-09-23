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
'''
Gaussian Naive Bayes (GaussianNB)
Decision Trees
Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
K-Nearest Neighbors (KNeighbors)
Stochastic Gradient Descent (SGDC)
Support Vector Machines (SVM)
Logistic Regression
'''
#from sklearn import tree                # decision tree node diagram graphviz
# insert here when confirmed
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# check these
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier # need gradient boosting
# --not in sk learn? -- neural networks

# metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import graphviz                     # decision tree node diagram

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
    ''''''
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
    def __init__(self, project, test_size=None):
        self.project    = ProjectData(project)
        if test_size!=None: self.test_size = test_size
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
        self.all        = pd.concat((self.X, self.y), axis=1)
        print("\tcreated 'all' dataframe, adding target as final column")
        self.features   = list(self.X.columns)
        self.label      = self.y.name
    def splitTrainTest(self):
        ''' use cross validation to split data into training and test datasets '''
        print("\nsplitting test and train data sets with {} test size and {} random state".format(self.test_size, self.random_state))
        self.Xtr, self.Xt, self.Ytr, self.Yt = model_selection.train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        print("\tXtrain / Xtest = {} / {}".format(len(self.Xtr), len(self.Xt)))
        print("\tYtrain / Ytest = {} / {}".format(len(self.Ytr), len(self.Yt)))

# === model ===
class MLModel(object):
    '''model wrapper for StudentData, post processing and split of training and test records'''
    def __init__(self, project):
        self.Xtr                        = project.Xtr
        self.Xt                         = project.Xt
        self.Ytr                        = project.Ytr
        self.Yt                         = project.Yt
    def train_classifier(self):
        '''Fits a classifier to the training data and time the effort''' # Start the clock, train the classifier, then stop the clock
        start                           = time()
        self.clf.fit(self.Xtr, self.Ytr)
        end                             = time()
        print("\t{:.4f} seconds to train model".format(end - start))
#    def predict_labels(clf, features, target):
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
    def setGaussianNB(self):
        ''''''
        self.clf = GaussianNB()
        self.train_classifier()
        self.predict_labels()
        self.clf.sigmas = sorted(zip(self.Xt.columns,self.clf.sigma_[0], self.clf.sigma_[1]), key=lambda x: x[1], reverse=True)  # sigma is variance of each parameter, theta is mean
        print("Gaussian - Naive Bayes sigmas for each input")
        for item in self.clf.sigmas: print("\t{:.4}\t{:.4}\t{}".format(item[1], item[2], item[0]))
    def setDecisionTree(self):
        ''''''
        self.clf = DecisionTreeClassifier()
        '''DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)[source]'''
        self.train_classifier()
        self.predict_labels()
        self.clf.importances = sorted(zip(self.Xt.columns, self.clf.feature_importances_), key=lambda x: x[1], reverse=True)
        print("decisionTree importances for each input")
        for item in self.clf.importances: print("\t{:.2}\t{}".format(item[1], item[0]))
#     def setBagging(self):   # find this library
#         '''Ensemble Methods, Bagging'''
# #        self.clf = xyz()
#         self.train_classifier()
#         self.predict_labels()
#     def setGradientBoosting(self):   # find this library
#         '''Ensemble Methods, Gradient Boosting'''
# #        self.clf = xyz()
#         self.train_classifier()
#         self.predict_labels()
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
    def setGradientDecent(self):
        '''Stochastic Gradient Descent'''
        pass
    # def setSVM(self):
    def setSVM(self, kernel='linear', C=1, gamma=0.1):
        self.kernel = kernel
        self.C      = C
        self.gamma  = gamma
        self.clf = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.train_classifier()
        self.predict_labels()
    # def plotData(self):
    #     '''callout to plot function'''
    #     prettyPicture(self.clf, self.project.X_test, self.project.y_test)
        print("support vectors {}".format(self.clf.support_vectors_))
        # get indices of support vectors
        print("support {}".format(self.clf.support_))
        # get number of support vectors for each class
        print("n_support {}".format(self.clf.n_support_))
    def setLogisticRegression(self, C=1e9):
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
        self.clf        = LogisticRegression(C=C)
        '''(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)[source]'''
        self.train_classifier()
        self.predict_labels()
        self.result          = pd.DataFrame(self.clf.coef_.transpose(),index=self.Xt.columns, columns=["coef"]) # create df with coefficients for each label
        self.result['abs']   = abs(self.result['coef'])
        pd.set_option('display.max_rows', 500)                      # show all features
        print(self.result.sort_values(by='abs', ascending=0))
