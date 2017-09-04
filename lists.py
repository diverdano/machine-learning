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
def testFirstN(count=1000000):
    return(sum(firstn(count)))

# === model object ===

class firstn(object):
    def __init__(self, n):
        self.n = n
        self.num, self.nums = 0, []

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            cur, self.num = self.num, self.num+1
            return cur
        else:
            raise StopIteration()


class ProjectData(object):
    ''' get and setup data '''
    infile  = 'projects.json'                    # should drop target/features from json? lift from data with pd.columns[:-1] & [-1]
    outfile = 'projects_backup.json'
    df_col  = 10
    tab     = r'\t'
    csv     = ','
    def __init__(self, project='credit_unions'):
        try:
            self.loadProjects()
            if project in self.projects.keys():
                self.desc       = project # if exists project in self.projects ...
                self.file       = self.projects[self.desc]['file']
                self.type       = self.projects[self.desc]['type']
                if self.type == 'tab'   : self.delim = self.tab
                else                    : self.delim = self.csv
                print(self.type, self.delim)
                self.loadData()
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
#     def readFile(self, type='csv'):
#         if      type == 'csv' : self.data = pd.read_csv(self.file, encoding = "ISO-8859-1")
#         elif    type == 'teb' : self.data = pd.read_table(self.file, encoding = "ISO-8859-1")
    def loadData(self):
        '''load data set as pandas.DataFrame'''
        self.data = pd.read_csv(self.file, encoding = "ISO-8859-1", sep=self.delim, thousands=',')       # c engine is faster, will use python if separater uses regular expression
        pd.set_option('display.width', None)                    # show columns without wrapping
        pd.set_option('display.max_columns', None)              # show all columns without elipses (...)
        pd.set_option('display.max_rows', self.df_col)               # show default number of rows for summary
        pd.options.display.float_format = '{:,.0f}'.format
        print("file loaded: {}".format(self.file))
        print("{} dataset has {} data points with {} variables each\n".format(self.desc, *self.data.shape))
        print("DataFrame Description (numerical attribute statistics)")
        print(self.data.describe())
        print("DataFrame, head")
        print(self.data.head())

class BankData(object):
    name        = "Bank Name / Holding Co Name"
    number      = "Bank ID"
    site        = 'Charter'
    main        = 'Bank Location'
    assets      = "Consol Assets (Mil $)"
    dom_assets  = "Domestic Assets (Mil $)"
    branches    = "Domestic Branches"
    def __init__(self, banks):
        self.banks          = ProjectData(banks).data[[self.number, self.name, self.site, self.main, self.branches, self.assets]]
        self.banks.columns = ['bankID','name','charter','location','branches', 'assets']
        self.banks.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    def sort_assets(self, df_col=None):
        pd.set_option('display.max_rows', df_col)               # show default number of rows for summary
        return self.banks.sort_values('assets', ascending=False)


class CreditUnionData(object):
    ''' base model object '''
    name        = "CU_NAME"
    number      = "CU_NUMBER"
    site        = 'SiteTypeName'
    main        = 'MainOffice'
    city        = "PhysicalAddressCity"
    state       = "PhysicalAddressStateCode"
    assets      = "ACCT_010"
#     merge_type  = 'inner'
    def __init__(self, cus, accounts):
        self.cus            = ProjectData(cus).data[[self.number, self.name, self.city, self.state, self.site, self.main]]
        self.accounts       = ProjectData(accounts).data[[self.number, self.assets]]
        self.merge()
    def merge(self):
        self.parents        = self.cus.loc[self.cus[self.main] == 'Yes']
        self.merged         = pd.merge(left=self.parents, right=self.accounts, how='inner', left_on=self.number, right_on=self.number)
        self.merged.columns = ['number','name','city','state', 'type', 'main', 'assets']
        self.CM             = self.merged.loc[self.merged['assets'] > 100000000]
    def sort_assets(self, df_col=None):
        pd.set_option('display.max_rows', df_col)               # show default number of rows for summary
        return self.CM.sort_values('assets', ascending=False)
