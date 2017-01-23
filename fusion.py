#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:29:19 2017

@author: valentin
"""

import pandas as pd
import os
os.chdir('/home/valentin/Documents/datacamp/')

path = '/home/valentin/Documents/datacamp/scrap_python/'

filenames = []
frame = []
for subdir, dirs, files in os.walk(path):
    print(subdir)
    print('--------------------------------------------------------')
    for file in files:
        filenames.append(file)
        if 'scrap_semaine_2' in str(subdir):
            file_path = str('scrap_python/scrap_semaine_2/'+ file)
        if 'scrap_semaine_3' in str(subdir):
            file_path = str('scrap_python/scrap_semaine_3/'+ file)
        print(file_path)
        frame.append(pd.read_csv(file_path, encoding = 'utf-8', dtype={'text':str}))

print(len(frame))

df = pd.DataFrame
df = pd.concat(frame)
df.shape[0]

ligne = 0
for i in range(len(frame)):
    ligne += frame[i].shape[0]
ligne


df.to_csv("fusion_semaine3_python.csv", header= True)

#test = pd.read_csv(os.path.join(path, 'fusion_semaine3_python.csv'), dtype={"text": str}, low_memory = False)


#list(test)[91]