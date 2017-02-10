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
tfidf = tkz.tfidf(tkz, text, index)
print(tfidf.shape)

bigrams = tkz.bigrams(tkz, text, index)
print(bigrams.shape)

tfidf = pd.concat([tfidf,bigrams], axis = 1)
print(tfidf.shape)

#CLASSIFICATION
X = np.asarray(tfidf)
del(tfidf)
del(bigrams)
Y = np.array(label)

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
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
score_func(Y_test,pred)
'''

#CROSS VALIDATION
from sklearn.metrics import make_scorer
f05_scorer = make_scorer(score_func)

from sklearn.model_selection import cross_val_score
estimators = [250,300,350,400, 450, 500]
'''
250
0.232154290504
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
'''
for i, n in enumerate(estimators):
    clf = XGBClassifier(learning_rate = 0.1, n_estimators = n)

    res.append(cross_val_score(clf, X, Y, scoring = f05_scorer, n_jobs=-1,
            cv = 2))
    print(n)
    print(np.mean(res[i]))
print(res)
'''
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
    scores = cross_val_score(clf, X, Y, cv=3, scoring=f05_scorer, n_jobs = -1)
    print(objective_function.n_iterations, \
        ": n_estimators = ", n_estimators, \
        "\tmax_depth = ", max_depth, \
        "fbeta_mean = ", np.mean(scores))
    return 1 - np.mean(scores)

from hyperopt import fmin as hyperopt_fmin
from hyperopt import tpe, hp, STATUS_OK, space_eval

objective_function.n_iterations = 0
best = hyperopt_fmin(objective_function,
        space=(hp.qloguniform('n_estimators', np.log(10), np.log(500), 10),
               hp.qloguniform('max_depth', np.log(2), np.log(100), 1)),
    algo=tpe.suggest,
    max_evals=10)

best

'''
probas_ = clf.predict_proba(X_test)
plt.plot(range(len(probas_)), probas_[:,1])
'''
