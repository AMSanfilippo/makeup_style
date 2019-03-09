#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir('your/code/dir/')

import pandas as pd
import numpy as np
import glob
import mca

data = pd.DataFrame()
for csv in glob.glob('data/*.csv'):
    to_append = pd.read_csv(csv)
    data = data.append(to_append)
    
data = data.reset_index(drop=True)

# variables btwn. which we want to determine correspondance (i.e. mu products)
products = ['blush', 'bronzer', 'brow_gel', 'brow_pencil', 'concealer',
       'eyeliner', 'eyeshadow', 'foundation', 'highlighter', 'lip_balm',
       'lip_gloss', 'lip_liner', 'lip_stain', 'lipstick', 'mascara','powder','primer']

# generate indicator matrix X using the correspondance variables
X = np.asmatrix(pd.get_dummies(data[products]).values)

# perform mca on indicator matrix
mca_out = mca.mca(X)
F = mca_out[0]
G = mca_out[1]
s = mca_out[2]

# select variables to project onto the principal axes ex-post as vertices
# want to see where these variables lie wrt the relationships identified 
# between "main" variables.
supplements = ['keyword']
X_supp = np.asmatrix(pd.get_dummies(data[supplements]).values)

# specify the coloring of our plotted points 
color_vec = data['score'].values

# specify labels for column vertices
labels = {}
for product in np.sort(products):
    labels[product] = list(np.unique(data[product]))

# specify labels for supplementary vertices 
labels_supp = {}
for supplement in np.sort(supplements):
    labels_supp[supplement] = list(np.unique(data[supplement]))

figpath = 'figures/'
figname = 'rp_allprod'

# plot row profiles, keyword vertices, scores from full (product/binary) mca
mca.plot_mca(figpath,figname,F,color=True,color_vec=color_vec,
         annotate_plt=True,X_supp=X_supp,s=s,labels_supp=labels_supp)

# interpretation:
# mua looks tend to be most closely associated with glam and natural styles.
# glam and natural looks are fairly closely related in terms of product usage,
# while beginner looks are starkly different.
# the highest-scoring posts tend to be more closely associated with a glam
# style, though some are associated with natural styles.

# consider other "cuts" of this data that may have interesting relationships

# focus on different features: eyes, skin, lips.
# for each feature, define level of focus as 'none','some','lots,' based on  
# number of products used.
focus = {'eyes':['brow_gel','brow_pencil','eyeliner','eyeshadow','mascara'],
         'face':['blush','bronzer','concealer','foundation','highlighter','powder'],
         'lips':['lip_gloss','lip_liner','lip_stain','lipstick']}

for f in focus.keys():
    
    focus_df = data[focus[f]]
    focus_df = focus_df.replace(['N','Y'],[0,1])
    focus_series = focus_df.sum(axis=1)
    data.loc[focus_series==0,f + '_focus'] = 'none'
    if f == 'eyes' or f == 'face':
        data.loc[(focus_series>0) & (focus_series<3),f + '_focus'] = 'some'
        data.loc[(focus_series>=3),f + '_focus'] = 'lots'
    else:
        data.loc[focus_series == 1,f + '_focus'] = 'some'
        data.loc[focus_series>=2,f + '_focus'] = 'lots'
        
# generate indicator matrix X using the correspondance variables
foci = ['eyes_focus','face_focus','lips_focus']
X = np.asmatrix(pd.get_dummies(data[foci]).values)

# perform mca on indicator matrix
mca_out = mca.mca(X)
F = mca_out[0]
G = mca_out[1]
s = mca_out[2]

# specify labels for column vertices
labels = {}
for f in np.sort(foci):
    labels[f] = list(np.unique(data[f]))
    
figname = 'rp_featurefocus'

# plot row profiles, keyword vertices, scores from full (product/binary) mca
mca.plot_mca(figpath,figname,F,G=G,color=False,color_vec=None,
         annotate_plt=True,labels=labels,X_supp=X_supp,s=s,labels_supp=labels_supp) 

# interpretation: 
# the contrast between glam/natural and beginner looks appears concentrated in
# the difference between minimal (beginner) and moderate/heavy (natural/glam)
# eye focus, and heavy (beginner) and minimal/moderate (natural/glam) lip focus