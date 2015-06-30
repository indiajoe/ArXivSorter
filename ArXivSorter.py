#!/usr/bin/env python
#This script is to sort ArXiv papers according to your interest using Machine Learning algorithms.

import os
import urllib
import itertools
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.linear_model import Lasso # ElasticNet
import numpy as np

# For the HTTP Server
from wsgiref.simple_server import make_server
from cgi import parse_qs, escape


# Location to save files
DIR = os.path.expanduser("~/ArXivSorter/")

class TopicSet:
    """ The class instant of each topic, one wishes to have independent instances of Archive Sorter. """
    def __int__(self,name,categories):
        """ Input : name str, the name of this Topic Archive Sorter.
            categories: input string tuple of ArXive Categories from which papers have to be retrieved."""
        self.name = name
        self.category = '%28cat:'+'+OR+cat:'.join(categories)+'%29+AND+'   # %28 and %29 are ( and ) in ascii
        self.InterestedArXivIDfile = os.path.join(DIR,self.name+'_Interested.list') #Filename of the file which contains list of Interested papers's ArXive Ids
        self.NotInterestedArXivIDfile = os.path.join(DIR,self.name+'_NotInterested.list') #Filename which contains list of Not Interested papers's ArXive Ids 
        self.NoofInt = 0           # No: of Interested Papers trained
        self.NoofNotInt = 0       # No: of Not Interested Papers trained
        self.vectorizer = TfidfVectorizer(min_df=1)

    def RetrieveArXivbyId(self,paperId):
        """ Retrieves the Paper's title, abstract, authors etc via ArXive's API in xml format 
        Input:  paperID is a string of ArXiv paper ids (multiple papers ids should be comma seperated)
        Output:  string, the full xml output of ArXiv API. """
        BasicURL = 'http://export.arxiv.org/api/query?'
        SearchQuery = 'id_list='+paperId
        url = BasicURL+SearchQuery
        print('Retrieving: '+url)
        return urllib.urlopen(url).read()
    

    def RetrieveArXivbyDate(self,StartDate=20140129,EndDate=20140131,MaxResults=50):
        """ Retrieves all the Paper's title, abstract, authors etc via ArXive's API in xml format 
        Input : StartDate  int, which gives the starting date of the search in format of YYYYMMDDHHMM 
        EndDate  int, which gives the endinging date of the search in format of YYYYMMDDHHMM 
        MaxResults int, which is a safe maximum limits of papers to retrieve 
        Output:  string, the full xml output of ArXiv API.    """
#      Category='%28cat:astro-ph.SR+OR+cat:astro-ph.EP%29+AND+'
    
        BasicURL = 'http://export.arxiv.org/api/query?'
        Maxcondition = '&max_results={0:d}'.format(MaxResults)
        SearchQuery = 'search_query='
        DateCondition = 'submittedDate:[{0:d}+TO+{1:d}]'.format(StartDate,EndDate)
        SearchQuery = SearchQuery+self.category+DateCondition+Maxcondition

        #Example:  http://export.arxiv.org/api/query?search_query=submittedDate:[200901130630+TO+200901131645]&max_results=200 
        url = BasicURL+SearchQuery
        print('Retrieving: '+url)
        return urllib.urlopen(url).read()

    def LoadIDsfromFile(self):
        """ Returns the list of Interested and Not Interested Papers Ids from corresponding files """
        with open(self.InterestedArXivIDfile,'r') as f :
            self.InterestedIds = [line.rstrip().split()[0] for line in f]
        with open(self.NotInterestedArXivIDfile,'r') as f :
            self.NotInterestedIds = [line.rstrip().split()[0] for line in f]
        return 

    def RetrieveDataToTrain(self,SampleData=None):
        """ Retrieves data from Archive to train the Model 
        Input: SampleData : feedparser output, if given will be used as Typical Data in."""
        self.Interested_data = feedparser.parse(self.RetrieveArXivbyId(','.join(self.InterestedIds)))
        self.NotInterested_data = feedparser.parse(self.RetrieveArXivbyId(','.join(self.NotInterestedIds)))
        self.Typical_data = SampleData or feedparser.parse(self.RetrieveArXivbyDate(StartDate=20140201,EndDate=20140207,MaxResults=100))
        
    def TrainOnData(self,SampleData=None):
        """ Trains the Model to find the Vector which reprosents Interested Paper and Also the Vector which represents Not interested papers 
        Input: SampleData : feedparser output, if given will be used as Typical Data in."""
        self.LoadIDsfromFile()
        self.RetrieveDataToTrain(SampleData=SampleData)

        Text_interested = (entry.title + entry.summary for entry in self.Interested_data.entries )   #String of Title and Summary text.
        Text_Notinterested = (entry.title + entry.summary for entry in self.NotInterested_data.entries ) 
        Text_Typical = (entry.title + entry.summary for entry in  self.Typical_data.entries)

        Data_2train = itertools.chain(Text_interested,Text_Notinterested,Text_2Rank) 
        Vectors_2train = self.vectorizer.fit_transform(Data_2train)  # Fitting TF-IDF of from data
        self.NoofInt = len(self.Interested_data.entries)
        self.NoofNotInt = len(self.NotInterested_data.entries)
        print('Trained with a sample Vectors of size '+str(Vectors_2train.shape))
        print('Sample size of Interested papers {0:d}'.format(self.NoofInt))
        print('Sample size of Not Interested papers {0:d}'.format(self.NoofNotInt))
        #Now we take mean of the Interested and NotInterested vectors to later use to find cosine similarity.
        self.InterestedVector=Vectors_2train[:self.NoofInt,:].mean(axis=0)
        self.NotInterestedVector=Vectors_2train[self.NoofInt:self.NoofInt+self.NoofNotInt,:].mean(axis=0)

    def ReturnRank(self,Data_2Rank):
        """ Returns the Rank indexes [-1 to 1] of the data based on cosine similarity with Interested and Notinterested vectors 
        Input: Data_2Rank, is feedparser output of the data to rank
        Output: Rank indexes [-1 to +1] of the input data based on cosine similarity. -1 is highest rank, and +1 is the lowest rank. """
        Text_2Rank = (entry.title + entry.summary for entry in  Data_2Rank.entries)
        Vectors_2Rank = self.vectorizer.transform(Text_2Rank)
        InterestedCosineRank = cosine_similarity(self.InterestedVector,Vectors_2Rank)[0]
        NotInterestedCosineRank = cosine_similarity(self.NotInterestedVector,Vectors_2Rank)[0]

        return NotInterestedCosineRank - InterestedCosineRank
        
    def RetrieveAndRank(self,StartDate=20140129,EndDate=20140131,MaxResults=200):
        """ Returns the Papers of input date range, and the rank to sort them 
        Input : StartDate  int, which gives the starting date of the search in format of YYYYMMDDHHMM 
        EndDate  int, which gives the endinging date of the search in format of YYYYMMDDHHMM 
        MaxResults int, which is a safe maximum limits of papers to retrieve 
        Output:  tuple(The Retrieved Full data in feedparser output format, List of Ranks of each entry)    """

        Data_2Rank = feedparser.parse(self.RetrieveArXivbyDate(StartDate=StartDate,EndDate=StartDate,MaxResults=MaxResults))
        return Data_2Rank,self.ReturnRank(Data_2Rank)




Mainhtml = """
<html>
<title>ArXive Sorter</title>
<body bgcolor="#ffffff">
</head>
<a name="top"></a> 
<center>
  <h1>ArXive Sorter</h1> <p> 
  <h3> <i> Your intelegent assistent to sort ArXive Papers to your taste </i> </h3> <br>
</center>
<a href="#Forms">[Go down to inputs]</a>
<!---All the Rank based Sorted Papers Starts Here --->
<p>
{SortedPapers}
</p>
<a href="#top">[Back to top]</a>
<hr size=5>
<!---All the Rank based Sorted Papers Ends Here --->
<!---Different input forms below --->
<a name="Forms"></a> 
<p>
{ToChooseTopic}
</p>
<p>
{ToChooseStartEndDate}
</p>
<p>
{ToCreateNewTopic}
</p>
<p>
{ToEditTopic}
</p>
<p>
{ToTrain}
</p>
<a href="#top">[Back to top]</a> <p>

Github Code Repository : <a href="http://indiajoe.github.io/ArXivSorter/">ArXivSorter</a>
<!---End of the document--->
   </body>
</html>


   <form method="post" action="parsing_post.wsgi">
      <p>
         Age: <input type="text" name="age" value="99">
         </p>
      <p>
         Hobbies:
         <input name="hobbies" type="checkbox" value="software"> Software
         <input name="hobbies" type="checkbox" value="tunning"> Auto Tunning
         </p>
      <p>
         <input type="submit" value="Submit">
         </p>
      </form>
   <p>
      Age: %s<br>
      Hobbies: %s
      </p>
   </body>
</html>
"""

#Template for html forms, which has to be inserted in Main html page
htmlform="""
   <form method="post" action="parsing_post.wsgi">
      <p>
      {0}
       </p>
      </form>
"""

FormsDict=dict()  #Dictionary of Forms to format Main html page

FormsDict[SortedPapers] = """ No Papers Loaded Yet!!. \n To Load Papers, Choose settings below.."""
FormsDict[ToChooseTopic] = """ """
FormsDict[ToChooseStartEndDate] = htmlform.format(""" Start Date: <input type="text" name="StartDate" placeholder="YYYYMMDD" >  End Date:  <input type="text" name="EndDate" placeholder="YYYYMMDD" >
  <input type="submit" value="Submit">
 """)
FormsDict[ToCreateNewTopic] = """ """
FormsDict[ToEditTopic] = """ """
FormsDict[ToTrain] = """ """

def ReturnhtmlForInput(inpdata):
    """ Returns the html page to display based on the user input from the page 
    Input: inpdata, the putput from parse_qs 
    Output: html string to display """
    

def application(environ, start_response):
    """ This part is coded loosely based on example in http://webpython.codepoint.net/wsgi_tutorial """
   # the environment variable CONTENT_LENGTH may be empty or missing
    try:
        request_body_size = int(environ.get('CONTENT_LENGTH', 0))
    except (ValueError):
        request_body_size = 0

    # When the method is POST the query string will be sent
   # in the HTTP request body which is passed by the WSGI server
   # in the file like wsgi.input environment variable.
    request_body = environ['wsgi.input'].read(request_body_size)
    inpdata = parse_qs(request_body)

    response_body = ReturnhtmlForInput(inpdata)

    age = d.get('age', [''])[0] # Returns the first age value.
    hobbies = d.get('hobbies', []) # Returns a list of hobbies.

   # Always escape user input to avoid script injection
    age = escape(age)
    hobbies = [escape(hobby) for hobby in hobbies]

    response_body = html % (age or 'Empty',
                            ', '.join(hobbies or ['No Hobbies']))

    status = '200 OK'
    
    response_headers = [('Content-Type', 'text/html'),
                  ('Content-Length', str(len(response_body)))]
    start_response(status, response_headers)

    return [response_body]

httpd = make_server('localhost', 8051, application)
httpd.serve_forever()





#-------------------------- End of the Code------ indiajoe@gmail.com

InterestedArXivIDfile = 'Interested.list'
NotInterestedArXivIDfile = 'NotInterested.list'

with open(InterestedArXivIDfile,'r') as f :
    InterestedIds = [line.rstrip() for line in f]
with open(NotInterestedArXivIDfile,'r') as f :
    NotInterestedIds = [line.rstrip() for line in f]

Interested_data = feedparser.parse(RetrieveArXivbyId(','.join(InterestedIds)))
NotInterested_data = feedparser.parse(RetrieveArXivbyId(','.join(NotInterestedIds)))
Data_2Rank = feedparser.parse(RetrieveArXivbyDate(StartDate=20140205,EndDate=20140207,MaxResults=100))

#Rank_2train=np.array([0.0]*len(InterestedIds) + [1.0]*len(NotInterestedIds))
Text_interested = (entry.title + entry.summary for entry in Interested_data.entries )  
Text_Notinterested = (entry.title + entry.summary for entry in NotInterested_data.entries ) 

Text_2Rank = (entry.title + entry.summary for entry in  Data_2Rank.entries)

Data_2train = itertools.chain(Text_interested,Text_Notinterested,Text_2Rank) 
vectorizer = TfidfVectorizer(min_df=1)
Vectors_2train = vectorizer.fit_transform(Data_2train)
print(Vectors_2train.shape)
NoOfIntIds = len(InterestedIds)
NoOfNotIntIds = len(NotInterestedIds)
InterestedVector = Vectors_2train[:NoOfIntIds,:].mean(axis=0)
NotInterestedVector = Vectors_2train[NoOfIntIds:NoOfIntIds+NoOfNotIntIds,:].mean(axis=0)

#Model=ElasticNet()
#Model=Lasso()
#Model.fit(Vectors_2train,Rank_2train)



Text_2Rank = (entry.title + entry.summary for entry in  Data_2Rank.entries)
Vectors_2Rank = vectorizer.transform(Text_2Rank)
InterestedCosineRank = cosine_similarity(InterestedVector,Vectors_2Rank)[0]
NotInterestedCosineRank = cosine_similarity(NotInterestedVector,Vectors_2Rank)[0]

FinalRank = NotInterestedCosineRank - InterestedCosineRank

#PredictedRank= Model.predict(Vector_2Rank)

SortOrder = np.argsort(FinalRank)
print(SortOrder)
for i in SortOrder:
    print('*'*30)
    print(FinalRank[i])
    print(Data_2Rank.entries[i].title)
    print(Data_2Rank.entries[i].authors)
#    print(Data_2Rank.entries[i].summary)
    print(Data_2Rank.entries[i].id.split('/abs/')[-1])

# for entry in data.entries:
#     print 'arxiv-id: %s' % entry.id.split('/abs/')[-1]
#     print 'Title:  %s' % entry.title
#         # feedparser v4.1 only grabs the first author
#     print 'First Author:  %s' % entry.authors


#titles=(entry.title for entry in data.entries)
#vectorizer = TfidfVectorizer(min_df=1)
#Vectors=vectorizer.fit_transform(titles)
#print(Vectors.toarray())
