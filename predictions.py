# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:46:48 2017

@author: Balaji

"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
stop=stopwords.words("english")
import pickle
from bs4 import BeautifulSoup as B
import re
filename = 'finalized_model.sav'
clf = pickle.load(open(filename, 'rb'))
vect=pickle.load(open('vect.sav','rb'))
q=["Last year my friend just couldn't stop talking about how hilarious and funny this was. Even some of my teachers had said good things about it. My friend showed me two episodes on his laptop and said, If you don't think this is funny, I don't know what it. Through the entire hour I laughed as many times as you've spoken while reading this review,zero. It still amazes me that hilarious shows such as Arrested Development get cancelled and this stays on air. For those who say this is  Smart Comedy how is this funny when every funny thing in this show is easy for even the most idiotic people to understand. Than in Arrested Development you can watch episodes over and over again and still find new bits of comedy that you hadn't seen before. Than back to this show where people claim the funniest thing is Sheldon dressed like a zebra making weird noises. What's so smart about that? The worst thing about this show is the Laugh Track. Now I don't mind laugh tracks if the show is funny but the laugh track just pops up at random places. A character will say something like, I want to be a robot. Than the laugh track kicks off, sounding like they just heard the funniest thing in the world. The whole series are based on socially awkward nerds who meet a normal girl and they try uselessly to fit in with society. How can an entire 5 seasons be focused on that? One person commented saying in 5 minutes of Arrested Development they get more laughs than they get in a season of Big Bang. I disagree, I think in 5 minutes of Arrested Development I get more laughs than I get in the entire series."]
def clean(review):
    text=B(review)
    text=text.get_text()
    text=re.sub("[^a-zA-Z]"," ",text)
    text=text.lower().split()
    text=[w for w in text if w not in stop]
    return(" ".join(text))




def test(q):
    #q=clean(q)
    q=vect.transform(q).toarray()
    pred=clf.predict(q)
    if pred==1:
        print('\n\tyou have given a positive review')
    else:
        print('\n\tyou have given a negative review')
       
        

q = input("Enter your review:\n ")
q=[q]
test(q)


