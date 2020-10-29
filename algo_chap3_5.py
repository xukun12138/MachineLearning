# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:05:33 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Administrator\\Desktop\\Data\\watermelon_3a.csv')

def calulate_wm():
    df1 = df[df.label == 1]
    df2 = df[df.label == 0]
    X1 = df1.values[:, 1:3]
    X0 = df2.values[:, 1:3]
    mean1 = np.array([np.mean(X1[:, 0]), np.mean(X1[:, 1])])
    mean0 = np.array([np.mean(X0[:, 0]), np.mean(X0[:, 1])])
    m1 = np.shape(X1)[0]
    sw = np.zeros(shape=(2, 2))
    for i in range(m1):
        xsmean = np.mat(X1[i, :] - mean1)
        sw += xsmean.transpose() * xsmean
    m0 = np.shape(X0)[0]
    for i in range(m0):
        xsmean = np.mat(X0[i, :] - mean0)
        sw += xsmean.transpose() * xsmean
    w = (mean0 - mean1) * (np.mat(sw).I)
    return w

def plot(w):
    dataMat = np.array(df[['density', 'ratio_sugar']].values[:, :])
    labelMat = np.mat(df['label'].values[:]).transpose()
    m = np.shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i, 0])
            ycord1.append(dataMat[i, 1])
        else:
            xcord2.append(dataMat[i, 0])
            ycord2.append(dataMat[i, 1])
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-0.2, 0.8, 0.1)
    y = np.array((-w[0, 0] * x) / w[0, 1])
    print(np.shape(x))
    print(np.shape(y))
    plt.sca(ax)
    plt.plot(x, y)  # gradAscent
    plt.xlabel('Density')
    plt.ylabel('Ratio_sugar')
    plt.title('LDA')
    plt.show()

w = calulate_wm()
plot(w)