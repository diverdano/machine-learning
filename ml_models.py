#!usr/bin/python

# === load libraries ===
# key libraries
import numpy as np
import pandas as pd
import simplejson as json

# data prep
from sklearn import model_selection
from sklearn.model_selection import KFold

# data sets
from sklearn.datasets import load_linnerud      # linear regression data set
from sklearn.datasets import load_digits


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
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# === model object ===

class Model(object):
    ''' base model object '''
    infile = 'ml_projects.json'
    def __init__(self, project='boston_housing'):
        if project in self.projects.keys():
            self.desc       = project # if exists project in self.projects ...
            self.file       = self.projects[self.desc]['file']
            self.target     = self.projects[self.desc]['target']
            self.features   = self.projects[self.desc]['features']
            self.loadData()
            self.prepData()
        else:   print('list of projects: ' + str(self.projects.keys()))
    def loadProjects(self):
        self.file_projects  = json.load(open(self.infile))
    def loadData(self):
        '''load data set'''
        self.data           = pd.read_csv(self.file)
        print("\n" + self.desc + " has {} data points with {} variables each\n".format(*self.data.shape))
        print(self.data.describe())
    def prepData(self):
        '''split out target and features'''
        self.target_data    = self.data[self.target]
        self.feature_data   = self.data.drop(self.target, axis = 1)


# === load & transform data ===

def loadData(file = '~/dev/machine-learning/projects/titanic_survival_exploration/titanic_data.csv'):
    '''load the dataset, default to titanic data set'''
    x = pd.read_csv(file)
    # Limit to numeric data
    x = x._get_numeric_data()
#     Separate the labels
    y = x['Survived']
#     Remove labels from the inputs, and age due to missing data
    del x['Age'], x['Survived']
    return {'features': x, 'labels': y}

def loadDataReg():
    '''load regression data sample'''
    linnerud_data = load_linnerud()
    x = linnerud_data.data
    y = linnerud_data.target
    return {'data': x, 'target': y}

def setup_data():
    '''setup data for learning curve plots'''
    digits = load_digits()
    x, y = digits.data, digits.target
    return (x,y,digits.data.shape[0])

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


#def plotData(data, title=None, xlabel=None, ylabel=None, grid=True, legend=True):
def plotData(plot_series, title=None, xlabel=None, ylabel=None, grid=True, legend=True):
    ''' plot data series '''
    plt.figure()
    if title == None:   title = 'Title of graph'
    plt.title(title)
    if xlabel == None:  xlabel = 'x axis label'
    plt.xlabel(xlabel)
    if ylabel == None:  ylabel = 'y axis label'
    plt.ylabel(ylabel)
    if grid == True:    plt.grid()
    if legend == True:  plt.legend(loc="best")
    plt.plot(data.index, data)
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 20)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None: plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean   = np.mean(train_scores, axis=1)
    train_scores_std    = np.std(train_scores, axis=1)
    test_scores_mean    = np.mean(test_scores, axis=1)
    test_scores_std     = np.std(test_scores, axis=1)
    print('train score mean & std: ', train_scores_mean, train_scores_std)
    print('test score mean & std : ', test_scores_mean, test_scores_std)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def plot_NB(n_iter=100):
    '''setup for Naive Bays learning curve'''
    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    estimator = GaussianNB()
    x,y,shape = setup_data()            # use data setup
    cv = cross_validation.ShuffleSplit(shape, n_iter=n_iter, test_size=0.2, random_state=0)
    print(cv)
    plot_learning_curve(estimator, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()

def plot_SVC(n_iter=10):
    '''setup for SVM RBF kernel, SVC is more expensive so do with lower number of CV iterations'''
    title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    x,y,shape = setup_data()            # use data setup
    cv = cross_validation.ShuffleSplit(shape, n_iter=n_iter, test_size=0.2, random_state=0)
    print(cv)
    estimator = SVC(gamma=0.001)
    plot_learning_curve(estimator, title, x, y, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()

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
    plot_learning_curve(estimator, title, X, y, (0.1, 1.01), cv=cv, scoring=score, n_jobs=4)
    plt.show()

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
    plot_learning_curve(estimator, title, X, y, (-0.1, 1.1), cv=cv, scoring=score, n_jobs=4)
    plt.show()
    
    
#     def plot_curve():
#         # YOUR CODE HERE
#         reg = DecisionTreeRegressor()
#         reg.fit(X,y)
#         print(reg.score(X,y))
# 
#         # TODO: Create the learning curve with the cv and score parameters defined above.
#     
#         # TODO: Plot the training and testing curves.
# 
#         # Show the result, scaling the axis for visibility
#         plt.ylim(-.1,1.1)
#         plt.show()
