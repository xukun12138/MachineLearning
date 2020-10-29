# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:51:17 2020

@author: Administrator
"""

import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\\Administrator\\Desktop\\Data\\watermelon_3a.csv')

arr = np.array([1] * len(df))
b = np.mat(df[['density', 'ratio_sugar']])
X = np.column_stack([b, arr])
Y = np.array(df['label']) 

beta = np.random.rand(3,1) #随机生成初始β值

def getP1(X,beta):
    m,n = X.shape
    P1 = []
    for i in range(m):
        P1.append((np.e ** np.dot(X[i],beta)[0,0])/(1+np.e ** np.dot(X[i],beta)[0,0]))
    return np.array(P1)

def getDbeta(X,Y,beta):
    P1 = getP1(X,beta)
    m,n = X.shape
    Dbeta = np.zeros((3,1))
    for i in range(m):
        Dbeta += X[i].T*(Y[i]-P1[i])
    return -Dbeta

def getD2beta(X,beta):
    P1 = getP1(X,beta)
    m, n = X.shape
    D2beta = np.zeros((3,3))
    for i in range(m):
        D2beta += np.dot(X[i].T,X[i])*P1[i]*(1-P1[i])
    return np.mat(D2beta)

while np.linalg.norm(-np.dot(getD2beta(X,beta).I,getDbeta(X,Y,beta)))>0.0001:
    beta1=beta-np.dot(getD2beta(X,beta).I,getDbeta(X,Y,beta))#书上的牛顿迭代公式
    beta=beta1
    
print(beta)#小于则输出最终结果