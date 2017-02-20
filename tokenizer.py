#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:31:45 2017

@author: valentin
"""

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import enchant
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



class Tokenizer:
    def __init__(self):
        pass

    def correct(self,word):
        d = enchant.Dict("fr_EU")
        try:
            word=d.suggest(word)[0]
        except Exception as e:
            pass
        return word

    def tokenize(self,text):
        text = text.lower()
        text=re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)',' ', text,flags=re.MULTILINE) #delete @
        text=re.sub(r'https?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE)
        text=re.sub(r'http?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE)
        text = re.sub('[éèêë]',"e",text)
        text = re.sub('[à]', "a", text)
        text = re.sub('[öô]', "o", text)
        text = re.sub('[îï]', "i", text)
        text = re.sub('[ùûü]', "u", text)
        text = re.sub('ç', "c", text)
        text = re.sub('[^A-Za-z0-9]+'," ",text) #on garde aussi les nombres
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if len(token) > 1:
                if re.search('[A-Za-z]', token):
                    filtered_tokens.append(token)
        return filtered_tokens

    def stem(self,tweet, stemmer):
        for i,token in enumerate(tweet):
            tweet[i] = stemmer.stem(token)
        return tweet

    def tfidf(self, tkz, text, index):
        vectorizer = TfidfVectorizer(preprocessor = None, tokenizer = tkz.tokenize,
                             stop_words = stopwords.words('french'))
        sparse = vectorizer.fit_transform(text).toarray()
        return pd.DataFrame(sparse, columns=vectorizer.get_feature_names(), index = index)

    def bigrams(self, tkz, text, index):
        vectorizer = TfidfVectorizer(preprocessor=None, tokenizer=tkz.tokenize,
                             stop_words=stopwords.words('french'), ngram_range = (2,2))

        vectorizer.fit(text)
        cols = [c for c in vectorizer.get_feature_names() if 'controleur' in c]

        sparse = vectorizer.transform(text.iloc[:text.shape[0]//3]).toarray()
        tfidf_bi = pd.DataFrame(sparse, columns=vectorizer.get_feature_names(),
                index = index[:len(index)//3])[cols]

        sparse = vectorizer.transform(text.iloc[text.shape[0]//3:(2*text.shape[0])//3]).toarray()
        tfidf_bi = pd.concat([tfidf_bi,pd.DataFrame(sparse, columns=vectorizer.get_feature_names(),
                index = index[len(index)//3:(2*len(index))//3])[cols]])
        sparse = vectorizer.transform(text.iloc[(2*text.shape[0])//3:]).toarray()
        return pd.concat([tfidf_bi,pd.DataFrame(sparse, columns=vectorizer.get_feature_names(),
                index = index[(2*len(index))//3:])[cols]])

    def counter(self, tkz, text, index):
        vectorizer = CountVectorizer(preprocessor = None, tokenizer = tkz.tokenize,
                             stop_words = stopwords.words('french'))
        sparse = vectorizer.fit_transform(text).toarray()
        return pd.DataFrame(sparse, columns=vectorizer.get_feature_names(), index = index)

    def count_bigrams(self, tkz, text, index):
        vectorizer = CountVectorizer(preprocessor=None, tokenizer=tkz.tokenize,
                             stop_words=stopwords.words('french'), ngram_range = (2,2))

        vectorizer.fit(text)
        cols = [c for c in vectorizer.get_feature_names() if 'controleur' in c]

        sparse = vectorizer.transform(text.iloc[:text.shape[0]//3]).toarray()
        tfidf_bi = pd.DataFrame(sparse, columns=vectorizer.get_feature_names(),
                index = index[:len(index)//3])[cols]

        sparse = vectorizer.transform(text.iloc[text.shape[0]//3:(2*text.shape[0])//3]).toarray()
        tfidf_bi = pd.concat([tfidf_bi,pd.DataFrame(sparse, columns=vectorizer.get_feature_names(),
                index = index[len(index)//3:(2*len(index))//3])[cols]])
        sparse = vectorizer.transform(text.iloc[(2*text.shape[0])//3:]).toarray()
        return pd.concat([tfidf_bi,pd.DataFrame(sparse, columns=vectorizer.get_feature_names(),
                index = index[(2*len(index))//3:])[cols]])
