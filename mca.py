#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# mca utilities

# perform mca on a indicator matrix X: each row is an obs, each column a level
# return row profiles and column vertices
def mca(X):
    
    N = X.sum()
    Z = X/N
    r = Z.sum(axis=1)
    c = Z.sum(axis=0)
    Dr = np.diagflat(r)
    Dc = np.diagflat(c)
    
    # compute svd
    nsqrt_Dr = np.linalg.inv(np.sqrt(Dr))
    nsqrt_Dc = np.linalg.inv(np.sqrt(Dc))
    S = np.matmul(np.matmul(nsqrt_Dr,(Z - np.outer(r,c))),nsqrt_Dc)
    U, s, V = np.linalg.svd(S, full_matrices=False)
    
    # compute row profiles 
    F = np.matmul(np.matmul(nsqrt_Dr,U),np.diagflat(s))
    
    # compute column vertices
    G = np.matmul(nsqrt_Dc,V)
    
    return [F,G,s]

# project supplementary variables onto the principal axes as column vertices
def project_supplements(X_supp,F,s):
    
    FD_inv = np.matmul(F,np.linalg.inv(np.diagflat(s)))
    I = np.size(X_supp,0)
    J = np.size(FD_inv,1)
    J_supp = np.size(X_supp,1)
    G_supp = np.zeros((J_supp,J))
    
    # G_supp is a J_supp x J matrix where entry (js,j) is the projection of the
    # jsth supplementary vertex onto the jth principal axis
    for i in range(J_supp):
        j = X_supp[:,i]
        j_sc = np.matmul(np.matmul(j.T,np.ones((I,1))),j.T)
        g_supp = np.matmul(j_sc,FD_inv)
        G_supp[i,:] = g_supp
        
    return G_supp

# re-scale supplement vertices to be same order of magnitude as rest of plot
def scale_supplements(supplement_vertices,o_plot):
    
    o_supp = int(math.log10(abs(supplement_vertices[:,:2]).max()))
    d = o_plot - o_supp
    if d != 0:
        supplement_vertices = supplement_vertices*(10**d)
    
    return supplement_vertices

# 2D plot row profiles, column vertices along principal axes of column profiles 
# F = row profile matrix, G = column vertex matrix
# color = binary; if yes, pass vector of values to use for color assignment
# annotate = binary; see below for labels
def plot_mca(figpath,figname,F,G=None,color=False,color_vec=None,annotate_plt=False,labels=None,X_supp=None,s=None,labels_supp=None):        
    
    plt.figure(1, figsize=(10, 6))
    
    row_profiles = np.array(F[:,:2])
    o_plot = int(math.log10(abs(row_profiles).max()))
    if color:
        plt.scatter(row_profiles[:,0],
                    row_profiles[:,1],
                    c=color_vec,cmap='YlOrRd',marker='o')
    else:
        plt.scatter(row_profiles[:,0],
                    row_profiles[:,1],marker='.',color=(0.528, 0.004, 0.284))
    if G is not None:
        column_vertices = np.array(G[:,:2])
        plt.scatter(column_vertices[:,0],column_vertices[:,1],marker='^')
        o_plot = int(math.log10(max(abs(row_profiles).max(),abs(column_vertices).max())))
    if X_supp is not None:
        G_supp = scale_supplements(project_supplements(X_supp,F,s),o_plot)
        supplement_vertices = G_supp[:,:2]
        plt.scatter(supplement_vertices[:,0],supplement_vertices[:,1],marker='+')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    
    if annotate_plt:
        if G is not None:
            notes = annotate(labels)
            for i, txt in enumerate(notes):
                plt.annotate(txt,(column_vertices[i,0],column_vertices[i,1]))
            
        if labels_supp is not None:
            notes_supp = annotate(labels_supp)
            for i, txt in enumerate(notes_supp):
                plt.annotate(txt,(supplement_vertices[i,0],supplement_vertices[i,1]))
                
    #plt.show()
    plt.savefig(figpath + figname + '.pdf')
    
            
# annotate the plot based on column names and levels
# labels should be a dict w/ key = variable name, values = list of levels
def annotate(labels):
    
    # generate the list of notes
    notes = []
    for var in labels.keys():
        notes += [var + '_' + i for i in labels[var]]
        
    return notes


            
