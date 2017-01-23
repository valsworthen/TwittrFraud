import pandas as pd
import numpy as np
import os
import re
import unicodedata
import enchant
from sklearn.feature_selection import RFE
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import FeatureAgglomeration
path = '//Users/charlesvaleyre/code/PycharmProjects/twitter'
filename = 'fusion_semaine3_python_tfid_1.csv'
import numbers
import collections
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def brutal_column():
    df = pd.read_csv(os.path.join(path, filename))
    print(df.shape)
    df = df[df.columns[df.sum() > 30]]
    print(df.shape)
    return df

#brutal_column()

def pca(n):
    df = brutal_column() 
    pca = PCA(n_components=n, svd_solver='full')
    fit = pca.fit(df)
    print(pca.explained_variance_ratio_) 
    return fit 

n= 20
pca(n)

def lda(n):
    df = brutal_column()
    
