# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:10:42 2017

@author: Balaji
"""

import pandas as pd
import nltk 
import sklearn
import re
from sklearn.naive_bayes import MultinomialNB
import numpy as n
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup as B
import pickle
data=pd.read_csv("C:/Users/Balaji/Downloads/labeledTrainData.tsv/labeledTrainData.tsv",header=0,delimiter="\t");
del data['id']
review=data['review']
sentiment=data['sentiment']
stop=stopwords.words("english")

#cleaningg
def clean(review):
    text=B(review)
    text=text.get_text()
    text=re.sub("[^a-zA-Z]"," ",text)
    text=text.lower().split()
    text=[w for w in text if w not in stop]
    return(" ".join(text))
    
final=[]
for i in range(0,len(review)):
    temp=clean(review[i])
    final.append(temp)
    
vect=CountVectorizer(analyzer="word",max_features=5000)
print(vect.fit(final))
x=vect.transform(final).toarray()
y=n.array(sentiment)
clf=MultinomialNB()
print(clf.fit(x,y))
#pred=clf.predict(x[20001:24999])
#print(sklearn.metrics.accuracy_score(y[20001:24999],pred))
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))
pickle.dump(vect, open('vect.sav', 'wb'))

