# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:01:33 2020

@author: Administrator
"""


import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


def set_ax_white(ax):
    ax.patch.set_facecolor("white")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    #ax.spines['bottom'].set_color('none')
    #ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')


path = r'C:\Users\Administrator\Desktop\Data\watermelon3_0a_Ch.txt'
data = pd.read_table(path, delimiter=' ', dtype=float)

X = data.iloc[:, [0]].values
y = data.iloc[:, 1].values

gamma = 10
C = 1

ax = plt.subplot()
set_ax_white(ax)
ax.scatter(X, y, color='C', label='data')

for gamma in [1, 10, 100, 1000]:
    svr = svm.SVR(kernel='rbf', gamma=gamma, C=C)
    svr.fit(X, y)

    ax.plot(np.linspace(0.2, 0.8), svr.predict(np.linspace(0.2, 0.8).reshape(-1, 1)),
            label='gamma={}, C={}'.format(gamma, C))
ax.legend(loc='upper left')
ax.set_xlabel('Density')
ax.set_ylabel('Sugar Content')

plt.rcParams['figure.dpi'] = 1000   #分辨率
plt.show()
