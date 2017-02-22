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
                 low_memory = False, usecols = ['text', 'label'])

#Trying to add more data
filename = 'set3_4000_labeled.csv'
tweets = pd.concat([tweets, pd.read_csv(os.path.join(path, filename), usecols = ['text', 'label'])], ignore_index = True)


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

#tfidf = pd.concat([tfidf,pd.Series(size, index = index)], axis = 1)
#print(tfidf.shape)
X['size'] = size
print(X.shape)
features = np.array(list(X))

#FEATURE SELECTION
X = np.asarray(X)

del(bigrams)
Y = np.array(label)

clf = XGBClassifier(learning_rate = 0.1, n_estimators = 200, nthread = -1)

import time
%time clf.fit(X, Y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.bar(range(200), importances[indices[0:200]])
plt.ylim([0,max(importances)])
plt.xlim([-1,200])
plt.show()

from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)

pd.Series(features[indices[0:152]]).to_csv('features_total.csv')

#CLASSIFICATION
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
    
from sklearn.metrics import make_scorer
f05_score = make_scorer(score_func)

from sklearn.model_selection import cross_val_score

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


estimators = [200,300,500,600,800,1000]
depth = [2,3,4,5,6,7,8,9,10]
child = [2,4,6,8,10,12]
gamma = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
sub = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
lambda_ = [0.8, 1,2,3,7]
rate = [0.01,0.05,0.1,0.2,0.3,0.6]

res = []
print('Début CV')
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 45)

for i, n in enumerate(estimators):
    clf = XGBClassifier(learning_rate = 0.1, n_estimators = n, max_depth = 3, min_child_weight= 1, gamma = 0, 
                        colsample_bytree= 1, subsample=1, reg_alpha=0, reg_lambda= 1, nthread=-1)
    print(n)
    start_step = time.time()
    res.append(cross_val_score(clf, X_new, Y, scoring = f05_score, n_jobs=1, cv = kfold))
    print('End step ' + str(time.time()-start_step))
    print(np.mean(res[i]))
print(res)

plt.scatter(estimators, np.mean(res, axis =1))
plt.show()
#0.445905721413

#test
clf = XGBClassifier(learning_rate = 0.1, n_estimators = 500, max_depth = 3, min_child_weight= 1, gamma = 0, 
                        colsample_bytree= 1, subsample=1, reg_alpha=0, reg_lambda= 1, nthread=-1)
scores = cross_val_score(clf, X, Y, scoring = f05_score, n_jobs=1, cv = kfold, verbose = 1)

#HYPEROPT
estimators = []
res = []
depth = []
child = []
def objective_function(x_int):
    print(x_int)
    objective_function.n_iterations += 1
    n_estimators, max_depth, min_child_weight = x_int
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_child_weight = int(min_child_weight)
    clf = XGBClassifier(learning_rate = 0.1, n_estimators=n_estimators, max_depth= max_depth, min_child_weight=min_child_weight, nthread = -1)
    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 7)
    start_step = time.time()
    scores = cross_val_score(clf, X_new, Y, cv=kfold, scoring=f05_score, n_jobs = 1)
    print('End step ' + str(time.time()-start_step))
    print(objective_function.n_iterations, \
        ": n_estimators = ", n_estimators, \
        "\tmax_depth = ", max_depth, \
        "\tmin_child_weight = ", min_child_weight, \
        "fbeta_mean = ", np.mean(scores))
    estimators.append(n_estimators)
    res.append(np.mean(scores))
    depth.append(max_depth)
    child.append(min_child_weight)
    
    steps.append(xint)
    return 1 - np.mean(scores)


from hyperopt import fmin as hyperopt_fmin
from hyperopt import tpe, hp, STATUS_OK, space_eval

objective_function.n_iterations = 0
start = time.time()
best = hyperopt_fmin(objective_function,
                    space=(hp.qloguniform('n_estimators', np.log(300), np.log(1000), 10),
                    hp.quniform('max_depth', 2, 15, 1),
                    hp.quniform('min_child_weight', 1, 10, 1)),
                    algo=tpe.suggest, max_evals=20)
print(time.time() - start)

plt.scatter(estimators, res)
plt.show()
plt.scatter(depth, res)
plt.show()
plt.scatter(child, res)
plt.show()

plt.scatter(range(20), res)
plt.show

best

'''
probas_ = clf.predict_proba(X_test)
plt.plot(range(len(probas_)), probas_[:,1])
'''
