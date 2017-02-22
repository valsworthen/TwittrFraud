import pandas as pd
import os

from xgboost import XGBClassifier
from tokenizer import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score


path = '/home/valentin/Documents/datacamp'
filename = 'set_labelised_10k_relabelisted.csv'
tweets = pd.read_csv(os.path.join(path, filename), dtype={"text": str},
                 low_memory = False, usecols = ['text', 'label', 'user.id_str'])

#Trying to add more data
filename = 'set3_4000_labeled.csv'
tweets = pd.concat([tweets, pd.read_csv(os.path.join(path, filename),
                    usecols = ['text', 'label', 'user.id_str'])], ignore_index = True)


text = tweets.text.drop_duplicates()
index = text.index
len(index)
label = tweets.label[index]

tkz = Tokenizer()

#PREPROCESS
X = tkz.counter(tkz, text, index)
print(X.shape)

bigrams = tkz.count_bigrams(tkz, text, index)
print(bigrams.shape)

X = pd.concat([X,bigrams], axis = 1)

size = []
for i in range(len(text)):
    size.append(len(tkz.tokenize(text.iloc[i])))


X['size'] = size
print(X.shape)
features = np.array(list(X))

X = np.asarray(X)
del(bigrams)
Y = np.array(label)

import time
clf = XGBClassifier(learning_rate = 0.1, n_estimators = 200, nthread = -1)
clf.fit(X, Y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
'''
plt.bar(range(200), importances[indices[0:200]])
plt.ylim([0,max(importances)])
plt.xlim([-1,200])
plt.show()
'''

from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf, prefit=True, threshold = 0.001)
X_new = model.transform(X)
del(X)

features_selected = features[indices[0:129]]
pd.Series(features_selected).to_csv('features_final.csv')

features_selected = pd.read_csv(os.path.join(path, 'features_final.csv')).values


#------------------------------------------------------------------------------

#UNLABELISED DATA
path = '/home/valentin/Documents/datacamp'
filename = 'set_non_labelised_12k.csv'

toBeLabelised = pd.read_csv(os.path.join(path, filename), dtype={"text": str},
                 low_memory = False, usecols = ['text', 'label', 'id_row'])
index_test = toBeLabelised.id_row

text_test = toBeLabelised.text
count_test = tkz.counter(tkz, text_test, index_test)

#add bigrams
bigrams_test = tkz.count_bigrams(tkz, text_test, index_test)

size_test = []
for i in range(len(text_test)):
    size_test.append(len(tkz.tokenize(text_test.iloc[i])))

count_test = pd.concat([count_test,bigrams_test], axis = 1)
print(count_test.shape)
count_test['size'] = size_test
print(count_test.shape)

del(bigrams_test)
del(tweets)
count_test = np.asarray(count_test[features_selected])
print(count_test.shape)

#------------------------------------------------------------------------------

#PREDICTION
clf = XGBClassifier(learning_rate = 0.1, n_estimators = 500, max_depth = 3, min_child_weight= 1, gamma = 0,
                        colsample_bytree= 1, subsample=1, reg_alpha=0, reg_lambda= 1, nthread=-1)
clf.fit(X_new, Y)

pred = clf.predict(count_test)

pd.concat([pd.Series(index_test, name = "id_row"),pd.Series(pred, name = "label")], axis = 1).to_csv('results.csv', header = True, index = False, sep = ';')
