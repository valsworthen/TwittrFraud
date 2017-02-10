import pandas as pd
import os

from xgboost import XGBClassifier
from tokenizer import Tokenizer
import numpy as np

path = '/home/valentin/Documents/datacamp'
filename = 'set_labelised_10k_relabelisted.csv'

tweets = pd.read_csv(os.path.join(path, filename), dtype={"text": str},
                 low_memory = False)

text = tweets.text.drop_duplicates()
index = text.index
len(index)
label = tweets.label[index]

tkz = Tokenizer()

#TF IDF
tfidf = tkz.tfidf(tkz, text, index)
print(tfidf.shape)

#add bigrams
bigrams = tkz.bigrams(tkz, text, index)
print(bigrams.shape)

tfidf = pd.concat([tfidf,bigrams], axis = 1)
print(tfidf.shape)

#------------------------------------------------------------------------------

#UNLABELISED DATA
path = '/home/valentin/Documents/datacamp'
filename = 'set_non_labelised_5k.csv'

toBeLabelised = pd.read_csv(os.path.join(path, filename), dtype={"text": str},
                 low_memory = False).text
index_test = toBeLabelised.index

tfidf_test = tkz.tfidf(tkz, toBeLabelised, index_test)

#add bigrams
bigrams_test = tkz.bigrams(tkz, toBeLabelised, index_test)

tfidf_test = pd.concat([tfidf_test,bigrams_test], axis = 1)
print(tfidf_test.shape)

appendTrain = [c for c in list(tfidf_test) if c not in list(tfidf)]
appendTest = [c for c in list(tfidf) if c not in list(tfidf_test)]

tfidf = pd.concat([tfidf, pd.DataFrame(np.zeros((tfidf.shape[0],
        len(appendTrain))), columns = appendTrain, index = index)], axis = 1)
tfidf_test = pd.concat([tfidf_test, pd.DataFrame(np.zeros((tfidf_test.shape[0],
    len(appendTest))), columns = appendTest, index = index_test)], axis = 1)

#------------------------------------------------------------------------------
#PREDICTION
tfidf = np.asarray(tfidf)
Y = np.asarray(label)
Y[np.where(Y==2)]=0

clf = XGBClassifier()
clf.fit(tfidf, Y.ravel())

tfidf_test = np.asarray(tfidf_test)

pred = clf.predict(tfidf_test)
