# In this exercise we'll examine a learner which has high variance, and tries to learn
# nonexistant patterns in the data.
# Use the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error. Use plt.plot() within the plot_curve function
# to create line graphs of the values.

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import KFold
from sklearn.metrics import explained_variance_score, make_scorer
import numpy as np

size = 1000
cv = KFold(size,shuffle=True)
score = make_scorer(explained_variance_score)

X = np.round(np.reshape(np.random.normal(scale=5,size=2*size),(-1,2)),2)
y = np.array([[np.sin(x[0]+np.sin(x[1]))] for x in X])

def plot_curve():
    # YOUR CODE HERE
    reg = DecisionTreeRegressor()
    reg.fit(X,y)
    print reg.score(X,y)
    estimator = DecisionTreeRegressor()
    train_sizes=np.linspace(.1, 1.0, 5)

    # TODO: Create the learning curve with the cv and score parameters defined above.
    train_sizes, train_scores, test_scores = learning_curve(reg, X, y, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))
#    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=reg.score(X,y), n_jobs=1, train_sizes=train_sizes)
#    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean   = np.mean(train_scores, axis=1)
    train_scores_std    = np.std(train_scores, axis=1)
    test_scores_mean    = np.mean(test_scores, axis=1)
    test_scores_std     = np.std(test_scores, axis=1)
    print('train score mean & std: ', train_scores_mean, train_scores_std)
    print('test score mean & std : ', test_scores_mean, test_scores_std)

    # TODO: Plot the training and testing curves.
    plt.grid()
#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")

    # Show the result, scaling the axis for visibility
    plt.ylim(-.1,1.1)
    plt.show()