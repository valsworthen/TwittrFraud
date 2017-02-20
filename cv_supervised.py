#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:47:42 2017

@author: valentin
"""

import pandas as pd
import os

from xgboost import XGBClassifier
from tokenizer import Tokenizer
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score


path = '/home/valentin/Documents/datacamp'
filename = 'set_labelised_10k_relabelisted.csv'

tweets = pd.read_csv(os.path.join(path, filename), dtype={"text": str},
                 low_memory = False)

text = tweets.text.drop_duplicates()
index = text.index
len(index)
label = tweets.label[index]

tkz = Tokenizer()

#PREPROCESS
#tfidf = tkz.tfidf(tkz, text, index)
#print(tfidf.shape)

count = tkz.counter(tkz, text, index)
print(count.shape)

#bigrams = tkz.bigrams(tkz, text, index)
#print(bigrams.shape)
bigrams = tkz.bigrams(tkz, text, index)
print(bigrams.shape)

#tfidf = pd.concat([tfidf,bigrams], axis = 1)
count = pd.concat([count,bigrams], axis = 1)

size = []
for i in range(len(text)):
    size.append(len(tkz.tokenize(text.iloc[i])))

#tfidf = pd.concat([tfidf,pd.Series(size, index = index)], axis = 1)
#print(tfidf.shape)

count = pd.concat([count,pd.Series(size, index = index, name = "size")], axis = 1)
print(count.shape)
features = np.array(list(count))

#FEATURE SELECTION
#X = np.asarray(tfidf)
X = np.asarray(count)
#del(tfidf)
del(count)
del(bigrams)
Y = np.array(label)

clf = XGBClassifier(learning_rate = 0.1, n_estimators = 200)
'''
IN : indices
Out[10]: array([13403,  3044,  5432, ...,  8906,  8905,     0])

In : importances[indices[0:10]]
Out : array([ 0.12689805,  0.06760665,  0.04555314,  0.04157628,  0.0307303 ,
        0.02928416,  0.02566884,  0.0253073 ,  0.0242227 ,  0.02205351], dtype=float32)
'''

import time
%time clf.fit(X, Y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print(features[indices[0:100]])

plt.bar(range(200), importances[indices[0:200]])
plt.ylim([0,max(importances)])
plt.xlim([-1,200])
plt.show()

from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)


#CLASSIFICATION

'''
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


clf.fit(X_train, Y_train)

#plot confusion matrix
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''
#Calcul fbeta score moyen
def score_func(y, y_pred, **kwargs):

    Y_test1 = y.copy()
    Y_test1[Y_test1 == 2] = 0
    pred1 = y_pred.copy()
    pred1[pred1==2] = 0
    f1 = fbeta_score(Y_test1, pred1, beta = 0.5)

    Y_test2 = y.copy()
    Y_test2[Y_test2 == 2] = 1
    pred2 = y_pred.copy()
    pred2[pred2==2] = 1
    f2 = fbeta_score(Y_test2, pred2, beta = 0.5)

    return np.average([f1,f2])

'''
class_names = np.unique(Y)


pred = clf.predict(X_test)
cnf_matrix = confusion_matrix(Y_test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, clas=ses=class_names,
                      title='Confusion matrix, without normalization')
score_func(Y_test,pred)
'''

#CROSS VALIDATION
from sklearn.metrics import make_scorer
f05_scorer = make_scorer(score_func)

from sklearn.model_selection import cross_val_score
estimators = [200,250,300]
'''
200
0.225481172054
250
0.238336451038
300
0.208889194582
350
0.20771403652
400
0.218339243864
450
0.172111948222
500
0.178991514537
'''

res = []
print('Début CV')

for i, n in enumerate(estimators):
    clf = XGBClassifier(learning_rate = 0.1, n_estimators = n)
    print(n)
    res.append(cross_val_score(clf, X_new, Y, scoring = f05_scorer, n_jobs=-1,
            cv = 2))
    print(np.mean(res[i]))
print(res)

#HYPEROPT

def objective_function(x_int):
    print(x_int)
    objective_function.n_iterations += 1
    n_estimators, max_depth = x_int
    n_estimators = int(n_estimators)
    print(n_estimators)
    max_depth = int(max_depth)
    print(max_depth)
    clf = XGBClassifier(learning_rate = 0.1, n_estimators=n_estimators,
                max_depth=max_depth)
    
    scores = cross_val_score(clf, X_new, Y, cv=3, scoring=f05_scorer)
    print(objective_function.n_iterations, \
        ": n_estimators = ", n_estimators, \
        "\tmax_depth = ", max_depth, \
        "fbeta_mean = ", np.mean(scores))
    return 1 - np.mean(scores)

from hyperopt import fmin as hyperopt_fmin
from hyperopt import tpe, hp, STATUS_OK, space_eval

objective_function.n_iterations = 0

%time best = hyperopt_fmin(objective_function,
                           space=(hp.qloguniform('n_estimators', np.log(10), np.log(500), 10),
                           hp.qloguniform('max_depth', np.log(2), np.log(100), 1)),
                           algo=tpe.suggest, max_evals=10)

best

'''
probas_ = clf.predict_proba(X_test)
plt.plot(range(len(probas_)), probas_[:,1])
'''
