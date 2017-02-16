#!usr/bin/python

# === load libraries ===

# key libraries
import numpy as np
import pandas as pd
import simplejson as json
from time import time

# data prep
from sklearn import model_selection

# models

# metrics
from sklearn.metrics import f1_score

# plot

# === data ===
def loadStudentData(file):
    return pd.read_csv(file)

# === test functions ===

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
        print("\n\ttarget attribute:\n\t\t'{}'".format(self.y.name))
        print("\n\tfeatures:")
        for index, item in enumerate(sorted(self.X.columns)): print("\t{}\t'{}'".format(index + 1, item))
    def preprocessData(self):
        ''' transpose objects to numerical data -> binary where appropriate '''
        # convert yes/no to 1/0
        if self.y.dtype == object:          self.y.replace(to_replace=['yes', 'no'], value=[1, 0], inplace=True)
        for col, col_data in self.X.iteritems():
            if col_data.dtype == object:    self.X[col].replace(to_replace=['yes', 'no'], value=[1, 0], inplace=True)
        # use separate for loop to complete in place changes before processing 'get_dummies'
        for col, col_data in self.X.iteritems():
            if col_data.dtype == object:    self.X = self.X.join(pd.get_dummies(col_data, prefix = col))
        # cast float64 to int64 for memory use and ease of reading (issue with get_dummies)
        for col in self.X.select_dtypes(['float64']):
            self.X[col] = self.X[col].astype('int64')
        # remove remaining object columns
        for col in self.X.select_dtypes(['O']):
            del self.X[col]
    def splitTrainTest(self):
        ''' use cross validation to split data into training and test datasets '''
        self.Xtr, self.Xt, self.Ytr, self.Yt = model_selection.train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
