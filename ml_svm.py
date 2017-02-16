#!usr/bin/python

# === load libraries ===
# key libraries
# import sys
import numpy as np
import pandas as pd
import random
import simplejson as json
# import copy
# import base64
# import subprocess


# data

# models
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import nltk

# metrics
from sklearn.metrics import accuracy_score

# plot
import matplotlib.pyplot as plt
import pylab as pl

# === data ===

# === test functions ===

# === helper functions ===

def prettyPicture(clf, X_test, y_test):
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0
    # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)
    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]
    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()
#    plt.savefig("test.png")

def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodestring(bytes)
    print(image_start+json.dumps(data)+image_end)

class ProjectData(object):
    ''' get and setup data '''
    infile  = 'ml_projects.json'
    outfile = 'ml_projects_backup.json'
    def __init__(self, project='boston_housing'):
        try:
            self.loadProjects()
            if project in self.projects.keys():
                if   project == 'reg'       : self.loadRegSample()
                elif project == 'lc'        : self.loadLearningCurveSample()
                elif project == 'terrain'   : self.makeTerrainData()
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
    def makeTerrainData(self, n_points=1000):
    ###############################################################################
    ### make the toy dataset
        self.data = {}
        random.seed(42)
        grade = [random.random() for ii in range(0,n_points)]
        bumpy = [random.random() for ii in range(0,n_points)]
        error = [random.random() for ii in range(0,n_points)]
        y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
        for ii in range(0, len(y)):
            if grade[ii]>0.8 or bumpy[ii]>0.8:
                y[ii] = 1.0
    ### split into train/test sets
        X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
        split = int(0.75*n_points)
        self.X_train = X[0:split]
        self.X_test  = X[split:]
        self.y_train = y[0:split]
        self.y_test  = y[split:]

# === model object ===

class Model(object):
#     kernel      = 'linear'
#     C           = 1
#     gamma       = 0.1
    accuracy    = None
    def __init__(self, project, kernel='linear', C=1, gamma=0.1):
        self.project    = ProjectData(project)
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

class ModelNB(Model):
    '''inherit from model, override create classifier with NB based'''
    def createClassifier(self):
        self.clf = GaussianNB()
    