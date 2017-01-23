#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:43:11 2017

@author: valentin
"""
import pandas as pd
import numpy as np
import os

path = '/home/valentin/Documents/datacamp'
#filename = 'fusion_semaine3_python_tfid_1.csv'
filename = 'FinalTF.csv'

my_data = pd.read_csv(os.path.join(path, filename), index_col = 0)
#my_data.TextTwitter

#factors = my_data[['https', 'hstg', 'arob']]
tweets = my_data['TextTwitter'].values
id = my_data['id'].values
my_data = my_data.drop(['TextTwitter', 'id'], axis = 1)
#my_data_nostem = pd.read_csv(os.path.join(path, filename), index_col = 0)

sum_word = list(np.sum(my_data.iloc[:,0:3000].values,axis=0))
sum_word += list(np.sum(my_data.iloc[:,3000:6000].values,axis=0))
sum_word += list(np.sum(my_data.iloc[:,6000:9000].values,axis=0))
sum_word += list(np.sum(my_data.iloc[:,9000:12000].values,axis=0))
sum_word += list(np.sum(my_data.iloc[:,12000:14185].values,axis=0))
len(sum_word)

words = [list(my_data)[i] for i in range(len(sum_word)) if sum_word[i] > 140]
len(words)

signif = my_data.copy()[words]
toBeClustered = signif.copy()
# clustering
from sklearn.cluster import KMeans
num_clusters = 7
km = KMeans(n_clusters=num_clusters).fit(toBeClustered)
clusters = km.labels_.tolist()
assignment=km.fit_predict(toBeClustered) 
signif['assignement']=assignment   

#------- VISU -------
signif['tweets'] = tweets

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
path = '/home/valentin/Documents/datacamp'
filename = 'twitter.png'
twitter_mask = np.array(Image.open(os.path.join(path,filename)))

stopwords = ['co', 'https', 'de', 'the', 'le', 'pas', 'rt']
wc = WordCloud(background_color="white", mask=twitter_mask, stopwords=stopwords)
for k in range(len(np.unique(assignment))):
    wc.generate(str(list(signif.tweets[assignment==k].dropna())))
    file = "wordcloud_"+str(k)+".png"
    wc.to_file(os.path.join(path, file))
    #print(signif.tweets[assignment==0])







plt.imshow(wc)
plt.axis("off")
plt.figure()
plt.show()


'''
path = '/home/valentin/Documents/datacamp'
filename = 'scrap_python/fusion_semaine3_python.csv'

signif['tweets'] = pd.read_csv(os.path.join(path, filename), dtype={"text": str}, 
                 low_memory = False).text.dropna().drop_duplicates()
'''