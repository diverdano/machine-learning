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


class RegVoters(object):
    '''get list of voters from voterecords.com using street (list) and cityst'''
    domain  = 'https://voterrecords.com/street/'
    cityst  = '-west+palm+beach-fl/'
    streets = [
        "bay+hill+dr",
        "blackwoods+ln",
        "buckhaven+ln",
        "carnegie+pl",
        "dunbar+ct",
        "eagles+xing",
        "gullane+ct",
        "keswick+way",
        "leeth+ct",
        "littlestone+ct",
        "marlamoor+ln",
        "northlake+blvd",
        "riverchase+run",
        "sanbourn+ct",
        "stonehaven+way",
        "torreyanna+cir"]
    urls_sets       = []
    test_urls       = []
    test_results    = []
    pages           = {}
    people          = []
    def __init__(self):
        self.testURLs()
        self.setupURLs()
    def testURLs(self):
        '''iterate streets, creating test urls, validate http status'''
        for street in self.streets:
            url = self.domain+street+self.cityst+"/1"
            self.test_urls.append(url)
            response    = util.requests.get(url)
            status      = response.status_code
            char_loc    = response.text.find('Page 1 of ')                        # find pages
            people_loc  = response.text.find(' people who live')
            if char_loc == -1   : pages = '1'
            else                : pages = response.text[char_loc:char_loc + 12].split('<')[0].split(' of ')[1]    # propably better way to do this
            people_count = response.text[people_loc - 10:people_loc].split(' are ')[1]
            self.pages[street] = {'pages':pages, 'people':people_count}
            print("\tstatus: {0} | {1} | {2} | {3}".format(str(status), pages, people_count, street))
        return self.pages
    def setupURLs(self):
        '''iterate streets, creating urls'''
        for street in self.streets:
            print(street)
            pages   = list(range(1,int(self.pages[street]['pages'])+1)) # assumes 9 pages of names for each street
            self.urls_sets.append([self.domain+street+self.cityst+str(page) for page in pages])
    def getVoters(self):
        self.people = []
        for i, urls in enumerate(self.urls_sets):
            for url in urls:
                print(url)
                response        = util.requests.get(url)
                self.people.append(util_data.pd.read_html(response.content))      # currently page contains summary table and list of voters
                # voters          = people[1].rename(columns=lambda x: x.strip()) # drop the summary table and strip space in column names
                # voters.drop(voters[[3]], axis=1, inplace=True)
                # self.people.append(voters)
#        self.people = util_data.pd.concat([self.people[item] for item in list(range(len(self.people)))])
