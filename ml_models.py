#!usr/bin/python

# === load libraries ===
# key libraries
import numpy as np
import pandas as pd
import simplejson as json

# data prep
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import sklearn.model_selection as curves
from sklearn.model_selection import GridSearchCV

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
#from projects.boston_housing import visuals as vs
#import visuals as vs

# === model object ===

class ProjectData(object):
    ''' get and setup data '''
    infile  = 'ml_projects.json'
    outfile = 'ml_projects_backup.json'
    def __init__(self, project='boston_housing'):
        try:
            self.loadProjects()
            if project in self.projects.keys():
                if project == 'reg'     : self.loadRegSample()
                elif project == 'lc'    : self.loadLearningCurveSample()
                else:
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
        '''load data set'''
        self.data           = pd.read_csv(self.file)
        print("file loaded: {}".format(self.file))
        print("{} dataset has {} data points with {} variables each\n".format(self.desc, *self.data.shape))
#        print("\n" + self.desc + " has {} data points with {} variables each\n".format(*self.data.shape))
        print(self.data.describe())
    def loadRegSample(self):
        ''' load regression sample dataset '''
        self.data           = load_linnerud()
#        self.X, self.y     = self.data.???, self.data.???    setup as dataframe
        print(self.data.DESCR)
    def loadLearningCurveSample(self):
        ''' load learning curve sample dataset '''
        self.data           = load_digits()
        print(self.data.DESCR)
    def prepData(self):
        '''split out target and features based on known column names in project meta data'''
        self.target_data    = self.data[self.target]
        self.feature_data   = self.data.drop(self.target, axis = 1)
#         if 'drop' in self.projects[self.desc]:                    if need to drop columns...
#             self.feature_data = self.feature_data.drop(self.)
#       once dataframe is created: df._get_numeric_data() to limit to numeric data only

client_data = [[5,17,15],[4,32,22],[8,3,12]]
clients = np.transpose(client_data)

class Model(object):
    ''' base model object '''
    test_size       = 0.20
    random_state    = 0
    n_splits        = 10
    params          = {'max_depth': list(range(1,11))}
    def __init__(self, project):
        self.project    = ProjectData(project)
        self.y          = self.project.target_data          # need to incorporate reg and lc data sets...
        self.X          = self.project.feature_data
        self.fit_model()
    def splitTrainTest(self):
        ''' use cross validation to split data into training and test datasets '''
#         self.test_size      = test_size
#         self.random_state   = random_state
        self.Xtr, self.Xt, self.Ytr, self.Yt = model_selection.train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
    def shuffleSplit(self):     # done inside of fit_model
        ''' use cross validation/shuffle to split data into training and test datasets '''
#        self.cv_sets = ShuffleSplit(self.X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
        self.cv_sets = ShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
#        self.ss_test, self.ss_train = self.cv_sets.split(self.X, self.y)
#        self.Xtr, self.Xt, Self.Ytr, self.Yt = self.cv_sets.split(self.X, self.y)
    def getR2(self, y_true, y_predict):
        ''' calculate performance (aka coefficient of determination, goodness of fit) '''
        r2_score   = r2_score(y_true, y_predict)
        return(r2_score)
    def fit_model(self):
        """ Performs grid search over the 'max_depth' parameter for a 
            decision tree regressor trained on the input data [X, y]. """
#     # Create cross-validation sets from the training data
#        self.cv_sets = ShuffleSplit(self.X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
        self.cv_sets = ShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)
        # TODO: Create a decision tree regressor object
        self.regressor = DecisionTreeRegressor()

        # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
#        self.params = {'max_depth':list(range(1,10))}      # set within class

        # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
#        self.scoring_fnc = make_scorer(self.getR2(y_test, y_train))
 
        # TODO: Create the grid search object
#        grid = GridSearchCV(self.regressor, self.params)     #score built into DecisionTreeRegressor
        grid = GridSearchCV(self.regressor, self.params, scoring = make_scorer(r2_score))

        # Fit the grid search object to the data to compute the optimal model
        grid = grid.fit(self.X, self.y)
        # Return the optimal model after fitting the data
#        return grid.best_estimator_
        self.best_est = grid.best_estimator_
    def viewRegPlot(self):
        ''' setup chart for plotting feature vs target variable '''
        reg = LinearRegression()
        # for feature in features: ...
        feature = self.project.features[0]
        feature_series = self.X[feature].reshape(-1,1)
        reg.fit(feature_series, self.y)
        # Visual aesthetics
        plt.legend(loc = 'lower right')
        plt.title("Regression Plot of {0} vs {1}".format(self.project.features[0], self.project.target))
        plt.xlabel(feature)
        plt.ylabel(self.project.target)
#        plt.ylim([-0.05,1.05])
        # Series plots
        plt.plot(feature_series, reg.predict(feature_series), color='red', linewidth=1)
        plt.scatter(feature_series, self.y, alpha=0.5, c=self.y)
        plt.show()
    def viewScatterPlots(self, newX, newY):
        ''' setup charts for plotting input features vs scatterplot of historical values '''
        for i, feat in enumerate(self.X.keys()):
            plt.scatter(self.X[feat], self.y)
            plt.scatter(newX[i], newY, color='red')
            plt.xlabel('feature {}'.format(feat))
            plt.show()
    def viewRegPlots(self):
        ''' setup charts for plotting features vs target variable '''
        fig = plt.figure(figsize=(10,7))

        # Create three different models based on max_depth
        for k, feature in enumerate(self.project.features):
            # Create a regression
            reg = LinearRegression(self.X[feature])
        
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
        fig.set_tight_layout(tight=tight)
        fig.show()


###

        
# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

# def fit_model(X, y):
#     """ Performs grid search over the 'max_depth' parameter for a 
#         decision tree regressor trained on the input data [X, y]. """
#     
#     # Create cross-validation sets from the training data
#     cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
# 
#     # TODO: Create a decision tree regressor object
#     regressor = None
# 
#     # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
#     params = {}
# 
#     # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
#     scoring_fnc = None
# 
#     # TODO: Create the grid search object
#     grid = None
# 
#     # Fit the grid search object to the data to compute the optimal model
#     grid = grid.fit(X, y)
# 
#     # Return the optimal model after fitting the data
#     return grid.best_estimator_

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


## === functions from boston_housing ProjectData

def ModelLearning(X, y, tight={'rect':(0,0,0.75,1)}):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """
    
    # Create 10 cross-validation sets for training and testing
#    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = plt.figure(figsize=(10,7))

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
#    ax.legend()
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