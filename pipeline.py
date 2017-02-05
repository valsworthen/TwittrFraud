import pandas as pd
import os
import enchant
from nltk.corpus import stopwords
import time
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizer import Tokenizer
from sklearn.metrics import roc_curve, auc
import numpy as np

path = '/home/valentin/Documents/datacamp'
filename = 'set_labelised_10k.csv'

tweets = pd.read_csv(os.path.join(path, filename), dtype={"text": str},
                 low_memory = False)

text = tweets.text.drop_duplicates()
index = text.index
len(index)
label = tweets.label[index]

#------------------------------------------------------------------------------
#TF IDF
french_stopwords = stopwords.words('french')
tkz = Tokenizer()
print('Start TfIdf')
start_time = time.time()
vectorizer = TfidfVectorizer(preprocessor = None, tokenizer = tkz.tokenize,
                             stop_words = french_stopwords)
tfidf = tkz.tfidf(vectorizer, text, index)
print('End TfIdf')
print("--- %s seconds ---" % (time.time() - start_time))
print(tfidf.shape)

#add bigrams
print('Start bigrams')
start_time = time.time()
vectorizer = TfidfVectorizer(preprocessor=None, tokenizer=tkz.tokenize,
                             stop_words=french_stopwords, ngram_range = (2,2))
bigrams = tkz.bigrams(vectorizer, text, index)
print('End bigrams')
print("--- %s seconds ---" % (time.time() - start_time))
print(bigrams.shape)

tfidf = pd.concat([tfidf,bigrams], axis = 1)
print(tfidf.shape)

#TRAINING
X = np.asarray(tfidf)
Y = np.asarray(label)
Y[np.where(Y==2)]=1

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

clf = XGBClassifier()
clf.fit(X_train, Y_train.ravel())

probas_ = clf.predict_proba(X_test)

# Compute false positive rate (FPR) and true positive rate (FPR)
fpr, tpr, _ = roc_curve(Y_test, probas_[:,1]) #comparaison de probas et y_test
# Compute area under the ROC curve
roc_auc = auc(fpr, tpr)
print('Auc  :')
print(roc_auc)
plt.plot(fpr, tpr, label="XGBoost test (auc=%.2f)" % roc_auc, lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver operating characteristic', fontsize=18)
plt.legend(loc="lower right", fontsize=14)

verif = (clf.predict(X_test) == Y_test)

'''
#------------------------------------------------------------------------------
#CORPUS: flatenning the list of tweets

allwords = flatten(tokens)
from collections import Counter
count = Counter(allwords)
print(count.most_common(100))

#allnormal = flatten(tokens)
corpus = list(set(allwords))

print("Corpus : ")
print(corpus)
print(len(corpus))

#------------------------------------------------------------------------------


#NGRAMS
from nltk import bigrams
from nltk import trigrams
my_bigrams = [list(bigrams(tweet)) for tweet in tokens]

new_bigrams = []
interesting = ['controle', 'controleur', 'attention', 'ligne', 'agent']
for word in interesting:
    for tweet in my_bigrams:
        new_bigrams.append([bigram for bigram in tweet if (word in bigram)])
#new_bigrams.append([bigram for bigram in tweet in my_bigrams if ('controleur' in bigram)])
new_bigrams
flatten = lambda l: [item for sublist in l for item in sublist]

bigrams = set(flatten(new_bigrams))
len(bigrams)


my_trigrams = [list(trigrams(tweet)) for tweet in tokens]
new_trigrams = []
for word in interesting:
    for tweet in my_trigrams:
        new_trigrams.append([trigram for trigram in tweet if (word in trigram)])

trigrams = set(flatten(new_trigrams))
len(trigrams)

'''
