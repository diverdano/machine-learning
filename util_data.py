#!usr/bin/env python

# == load libraries ==

# key libraries
import pandas as pd
import simplejson as json
import csv                                      # for csv sniffer
import logging

# data sets
from sklearn.datasets import load_linnerud      # linear regression data set
from sklearn.datasets import load_digits        # learning curve data set
import quandl

# == set logging ==
logger = logging.getLogger(__name__)
#logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

# == data ==

def getQuandl(symbol):
    '''access quandl data via api
    examples:
        vix futures: "CHRIS/CBOE_VX1", "CHRIS/CBOE_VX2"
        s&p 500 e-mini: "CME/ESU2018"'''
    return quandl.get(symbol)

def mergeDFs(df1, df2, df1_cols, df2_cols, join_type='outer'):
    '''merge df1 and df2 on df1_columns (list) and df2_columns (list) using join type'''
    df3 = pd.merge(df1, df2, left_on=df1_cols, right_on=df2_cols, how=join_type)
    return df3

def isolateMissing():
    '''isolate columns that aren't mapping'''
    return df3[df3.isnull().any(axis=1)][['name','company']]

def getFirstLast(name):
    '''separates the first and last names'''
    if isinstance(name, str):
        name        = name.split()
    else:
        name    = ('na','na')
    return{"firstname":name[0], "lastname":name[-1]}
    # firstName   = name[0]
    # lastName    = name[-1]
    # # now figure out the first initial, we're assuming that if it has a dot it's an initialized name, but this may not hold in general
    # if "." in firstName:
    #     firstInitial = firstName
    # else:
    #     firstInitial = firstName[0] + "."
    # lastName = name[2]
    # return {"FirstName":firstName, "FirstInitial":firstInitial, "LastName": lastName}

# == test functions ==
# def SniffDelim(file):
#     '''use csv library to sniff delimiters'''
#     with open(file, 'r') as infile:
#         dialect = csv.Sniffer().sniff(infile.read())
#     return dict(dialect.__dict__)

# == data ==

def loadRegSample(self):
    ''' load regression sample dataset '''
    self.data           = load_linnerud()
    logger.info(self.data.DESCR)
def loadLearningCurveSample(self):
    ''' load learning curve sample dataset '''
    self.data           = load_digits()
    logger.info(self.data.DESCR)

def makeTerrainData(self, n_points=1000):
    '''make the toy dataset'''
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
    self.data['X_train'] = X[0:split]
    self.data['X_test']  = X[split:]
    self.data['y_train'] = y[0:split]
    self.data['y_test']  = y[split:]
#     grade_sig = [self.data['X_train'][ii][0] for ii in range(0, len(self.data['X_train'])) if self.data['y_train'][ii]==0]
#     bumpy_sig = [self.data['X_train'][ii][1] for ii in range(0, len(self.data['X_train'])) if self.data['y_train'][ii]==0]
#     grade_bkg = [self.data['X_train'][ii][0] for ii in range(0, len(self.data['X_train'])) if self.data['y_train'][ii]==1]
#     bumpy_bkg = [self.data['X_train'][ii][1] for ii in range(0, len(self.data['X_train'])) if self.data['y_train'][ii]==1]
# #         training_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
# #                 , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}
#     grade_sig = [self.data['X_test'][ii][0] for ii in range(0, len(self.data['X_test'])) if self.data['y_test'][ii]==0]
#     bumpy_sig = [self.data['X_test'][ii][1] for ii in range(0, len(self.data['X_test'])) if self.data['y_test'][ii]==0]
#     grade_bkg = [self.data['X_test'][ii][0] for ii in range(0, len(self.data['X_test'])) if self.data['y_test'][ii]==1]
#     bumpy_bkg = [self.data['X_test'][ii][1] for ii in range(0, len(self.data['X_test'])) if self.data['y_test'][ii]==1]
# #         test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
# #                 , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}
# #         return X_train, y_train, X_test, y_test

# == data object ==

class ProjectData(object):
    ''' get and setup data '''
    infile          = 'ml_projects.json'                # should drop target/features from json? lift from data with pd.columns[:-1] & [-1]
    outfile         = 'ml_projects.json'                # write to same file, use git to manage versions
    df_col          = 10
    def __init__(self, project='boston_housing', file=None):
        if file:
            self.desc           = 'file used'
            self.file           = file
            self.loadData()
        else:
            try:
                self.loadProjects()
                if project in self.projects.keys():
                    self.desc       = project # if exists project in self.projects ...
                    self.file       = self.projects[self.desc]['file']
                    try:
                        self.loadData()
                    except Exception as e:
                        logger.error("issue loading data: ", exc_info=True)
#                        print("issue loading data: {}".format(e))
                        return
                    if all (k in self.projects[self.desc] for k in ('target', 'features')):
                        self.target     = self.projects[self.desc]['target']        # make y or move this to data, or change reg & lc samples?
                        self.features   = self.projects[self.desc]['features']      # make X or move this to data, or change reg & lc samples?
                        self.prepData()
                        self.preprocessData()
                    logger.warn("'target' and 'features' need to be specified for prepping model data")
#                    else: print("\n\t'target' and 'features' need to be specified for prepping model data")
                else:
                    logger.warn('"{}" project not found; list of projects:\n'.format(project))
                    logger.warn("\t" + "\n\t".join(sorted(list(self.projects.keys()))))
            except Exception as e: # advanced use - except JSONDecodeError?
                logger.error("issue reading project file:", exc_info=True)
#                print('issue reading project file: {}'.format(e))
    def loadProjects(self):
        ''' loads project meta data from file and makes backup '''
        with open(self.infile) as file: self.projects  = json.load(file)
    def saveProjects(self):
        ''' saves project meta detail to file '''
        with open(self.outfile, 'w') as outfile: json.dump(self.projects, outfile, indent=4)
    def loadData(self):
        '''check file delimiter and load data set as pandas.DataFrame'''
        try:
            logger.info('\n\tchecking delimiter')
            delimiter = self.sniffDelim()['delimiter']
            logger.info('\tdelimiter character identified: {}'.format(delimiter))
            try:
                self.data = pd.read_csv(self.file, sep=delimiter)
                logger.info("\tfile loaded")
                self.summarizeData()
            except UnicodeDecodeError:
                logger.error('\tunicode error, trying latin1')               # implement logging
                # self.data           = pd.read_csv(self.file, encoding='latin1')
                self.data           = pd.read_csv(self.file, encoding='latin1', sep=delimiter) # including sep in this call fails...
                logger.info("\tfile loaded")
                self.summarizeData()
        except Exception as e:
            raise e
#            print(e)
    def summarizeData(self):
        '''separate data summary/reporting'''
        pd.set_option('display.width', None)                    # show columns without wrapping
        pd.set_option('display.max_columns', None)              # show all columns without elipses (...)
        pd.set_option('display.max_rows', self.df_col)          # show default number of rows for summary
        logger.info("\n\t{} dataset has {} data points with {} variables each\n".format(self.desc, *self.data.shape))
        logger.info("\n\tDataFrame Description (numerical attribute statistics)\n")
        logger.info(self.data.describe())
        logger.info("\n\tDataFrame, head\n")
        logger.info(self.data.head())
        logger.info("\n\t{} Data Summary\n".format(self.desc))
        logger.info("\t{}\trecords".format(len(self.data.index)))
        logger.info("\t{}\tfeatures\n".format(len(self.data.columns)))
        for index, item in enumerate(sorted(self.data.columns)): logger.info("\t{}\t'{}'".format(index + 1, item))
    def prepData(self):
        '''split out target and features based on known column names in project meta data'''
        self.y              = self.data[self.target]
        self.X              = self.data.drop(self.target, axis = 1)
    def preprocessData(self):
        '''transpose objects to numerical data -> binary where appropriate '''
        # convert yes/no to 1/0
        logger.info("\n\tprocessing target/y variable")
        logger.info("\n\tpreprocessing X & y, inputs and target values, replacing yes/no with 1/0")
        if self.y.dtype == object:          self.y.replace(to_replace=['yes', 'no'], value=[1, 0], inplace=True)
        logger.info("\t\ty (target) values completed")
        for col, col_data in self.X.iteritems():
            if col_data.dtype == object:    self.X[col].replace(to_replace=['yes', 'no'], value=[1, 0], inplace=True)
        # use separate for loop to complete in place changes before processing 'get_dummies'
        logger.info("\t\tX (input) values completed")
        for col, col_data in self.X.iteritems():
            if col_data.dtype == object:    self.X = self.X.join(pd.get_dummies(col_data, prefix = col))
        logger.info("\tconverted categorical variable into dummy/indicator variables")
        # cast float64 to int64 for memory use and ease of reading (issue with get_dummies)
        for col in self.X.select_dtypes(['float64']):
            self.X[col] = self.X[col].astype('int64')
        logger.info("\tconverted float to integer")
        # if all else fails: _get_numeric_data()       # limit to numeric data
        # remove remaining object columns
        for col in self.X.select_dtypes(['O']):
            del self.X[col]
        logger.info("\tremoved initial columns, now that they have been converted")
        self.all        = pd.concat((self.X, self.y), axis=1)
        logger.info("\tcreated 'all' dataframe, adding target as final column")
        self.features   = list(self.X.columns)
        self.label      = self.y.name
        logger.info("\n\tTarget/Label Description (numerical attribute statistics)\n")
        logger.info(self.y.describe())
    def sniffDelim(self):
        '''helper functionuse csv library to sniff delimiters'''
        with open(self.file, 'r') as infile:
            dialect = csv.Sniffer().sniff(infile.read())
        return dict(dialect.__dict__)
