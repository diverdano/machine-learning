#!usr/bin/env python

# == load libraries ==
import util
import util_data
import util_plot

# == load data ==
# s&p 500 symbols
def getSP500():
    '''use pandas to parse wikipedia S&P 500 web pages'''
    url         = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response    = util.requests.get(url)
    result      = util_data.pd.read_html(response.content)      # currently page contains "component stocks" & "recent changes"
#     for df in result:
#         df.columns  = df.iloc[0]                                # first row contains the column names
# #        df.reindex(df.index.drop(0))
    return result
