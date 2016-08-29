print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import KFold
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.linear_model import LinearRegression




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
#     print('train sizes: ', train_sizes)
#     print('train scores: ', train_scores)
#     print('test scores: ', test_scores)
    train_scores_mean   = np.mean(train_scores, axis=1)
    train_scores_std    = np.std(train_scores, axis=1)
    test_scores_mean    = np.mean(test_scores, axis=1)
    test_scores_std     = np.std(test_scores, axis=1)
    print('train score mean & std: ', train_scores_mean, train_scores_std)
    print('test score mean & std : ', test_scores_mean, test_scores_std)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def setup_data():
    '''setup data for learning curve plots'''
    digits = load_digits()
    x, y = digits.data, digits.target
    return (x,y,digits.data.shape[0])


def plot_NB():
    '''setup for Naive Bays learning curve'''
    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    estimator = GaussianNB()
    x,y,shape = setup_data()
    cv = cross_validation.ShuffleSplit(shape, n_iter=100,
                                       test_size=0.2, random_state=0)
    print(cv)
    plot_learning_curve(estimator, title, x, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()

def plot_SVC():
    '''setup for SVM RBF kernel'''
    title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    x,y,shape = setup_data()
    cv = cross_validation.ShuffleSplit(shape, n_iter=10,
                                   test_size=0.2, random_state=0)
    print(cv)
    estimator = SVC(gamma=0.001)
    plot_learning_curve(estimator, title, x, y, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()

def plot_LR():
    '''setup for KFold '''
    title = "Learning Curves (Linear Regression)"
    X = np.reshape(np.random.normal(scale=2,size=1000),(-1,1))
    y = np.array([[1 - 2*x[0] +x[0]**2] for x in X])
#    shape = x.shape
    cv = KFold(1000,shuffle=True)
    print(cv)
    score = make_scorer(explained_variance_score)

#     reg = LinearRegression()
#     reg.fit(x,y)
#     print(reg.score(x,y))
#     estimator = reg
    estimator = LinearRegression()
#     train_sizes, train_scores, test_scores = learning_curve(estimator,
#         X,y,cv=cv,scoring=score,train_sizes=np.linspace(.1, 1.0, 5))

    plot_learning_curve(estimator, title, X, y, (0.1, 1.01), cv=cv, scoring=score, n_jobs=4)
    plt.show()



