# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 21:05:36 2020

@author: Administrator
"""


import numpy as np
import copy
import pandas as pd
from sklearn import datasets #5.6需要
from matplotlib import pyplot as plt

def xavier_initializer(layer_dims_, seed=16):
    np.random.seed(seed)

    parameters_ = {}
    num_L = len(layer_dims_)
    for l in range(num_L - 1):
        temp_w = np.random.randn(layer_dims_[l + 1], layer_dims_[l]) * np.sqrt(1 / layer_dims_[l])
        temp_b = np.zeros((1, layer_dims_[l + 1]))

        parameters_['W' + str(l + 1)] = temp_w
        parameters_['b' + str(l + 1)] = temp_b

    return parameters_


def he_initializer(layer_dims_, seed=16):
    np.random.seed(seed)

    parameters_ = {}
    num_L = len(layer_dims_)
    for l in range(num_L - 1):
        temp_w = np.random.randn(layer_dims_[l + 1], layer_dims_[l]) * np.sqrt(2 / layer_dims_[l])
        temp_b = np.zeros((1, layer_dims_[l + 1]))

        parameters_['W' + str(l + 1)] = temp_w
        parameters_['b' + str(l + 1)] = temp_b

    return parameters_


def cross_entry_sigmoid(y_hat_, y_):
    '''
    计算在二分类时的交叉熵
    :param y_hat_:  模型输出值
    :param y_:      样本真实标签值
    :return:
    '''

    m = y_.shape[0]
    loss = -(np.dot(y_.T, np.log(y_hat_)) + np.dot(1 - y_.T, np.log(1 - y_hat_))) / m

    return np.squeeze(loss)


def cross_entry_softmax(y_hat_, y_):
    '''
    计算多分类时的交叉熵
    :param y_hat_:
    :param y_:
    :return:
    '''
    m = y_.shape[0]
    loss = -np.sum(y_ * np.log(y_hat_)) / m
    return loss


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def relu(z):
    a = np.maximum(0, z)
    return a


def softmax(z):
    z -= np.max(z)  # 防止过大，超出限制，导致计算结果为 nan
    z_exp = np.exp(z)
    softmax_z = z_exp / np.sum(z_exp, axis=1, keepdims=True)
    return softmax_z


def sigmoid_backward(da_, cache_z):
    a = 1 / (1 + np.exp(-cache_z))
    dz_ = da_ * a * (1 - a)
    assert dz_.shape == cache_z.shape
    return dz_


def softmax_backward(y_, cache_z):
    #
    a = softmax(cache_z)
    dz_ = a - y_
    assert dz_.shape == cache_z.shape
    return dz_


def relu_backward(da_, cache_z):
    dz = np.array(da_, copy=True)
    dz[cache_z <= 0] = 0
    assert (dz.shape == cache_z.shape)

    return dz


def update_parameters_with_gd(parameters_, grads, learning_rate):
    L_ = int(len(parameters_) / 2)

    for l in range(1, L_ + 1):
        parameters_['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters_['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters_


def update_parameters_with_sgd(parameters_, grads, learning_rate):
    L_ = int(len(parameters_) / 2)

    for l in range(1, L_ + 1):
        parameters_['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters_['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters_


def initialize_velcoity(paramters):
    v = {}

    L_ = int(len(paramters) / 2)

    for l in range(1, L_ + 1):
        v['dW' + str(l)] = np.zeros(paramters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(paramters['b' + str(l)].shape)

    return v


def update_parameters_with_sgd_momentum(parameters, grads, velcoity, beta, learning_rate):
    L_ = int(len(parameters) / 2)

    for l in range(1, L_ + 1):
        velcoity['dW' + str(l)] = beta * velcoity['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
        velcoity['db' + str(l)] = beta * velcoity['db' + str(l)] + (1 - beta) * grads['db' + str(l)]

        parameters['W' + str(l)] -= learning_rate * velcoity['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * velcoity['db' + str(l)]

    return parameters, velcoity


def initialize_adam(paramters_):
    l = int(len(paramters_) / 2)
    square_grad = {}
    velcoity = {}
    for i in range(l):

        for i in range(l):
            square_grad['dW' + str(i + 1)] = np.zeros(paramters_['W' + str(i + 1)].shape)
            square_grad['db' + str(i + 1)] = np.zeros(paramters_['b' + str(i + 1)].shape)
            velcoity['dW' + str(i + 1)] = np.zeros(paramters_['W' + str(i + 1)].shape)
            velcoity['db' + str(i + 1)] = np.zeros(paramters_['b' + str(i + 1)].shape)
        return velcoity, square_grad


def update_parameters_with_sgd_adam(parameters_, grads_, velcoity, square_grad, epoch, learning_rate=0.1, beta1=0.9,
                                    beta2=0.999, epsilon=1e-8):
    l = int(len(parameters_) / 2)

    for i in range(l):
        velcoity['dW' + str(i + 1)] = beta1 * velcoity['dW' + str(i + 1)] + (1 - beta1) * grads_['dW' + str(i + 1)]
        velcoity['db' + str(i + 1)] = beta1 * velcoity['db' + str(i + 1)] + (1 - beta1) * grads_['db' + str(i + 1)]

        vw_correct = velcoity['dW' + str(i + 1)] / (1 - np.power(beta1, epoch))         # 这里是对迭代初期的梯度进行修正
        vb_correct = velcoity['db' + str(i + 1)] / (1 - np.power(beta1, epoch))

        square_grad['dW' + str(i + 1)] = beta2 * square_grad['dW' + str(i + 1)] + (1 - beta2) * (
                    grads_['dW' + str(i + 1)] ** 2)
        square_grad['db' + str(i + 1)] = beta2 * square_grad['db' + str(i + 1)] + (1 - beta2) * (
                    grads_['db' + str(i + 1)] ** 2)

        sw_correct = square_grad['dW' + str(i + 1)] / (1 - np.power(beta2, epoch))
        sb_correct = square_grad['db' + str(i + 1)] / (1 - np.power(beta2, epoch))

        parameters_['W' + str(i + 1)] -= learning_rate * vw_correct / np.sqrt(sw_correct + epsilon)
        parameters_['b' + str(i + 1)] -= learning_rate * vb_correct / np.sqrt(sb_correct + epsilon)

    return parameters_, velcoity, square_grad


def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")  #背景颜色
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')


def plot_costs(costs, labels, colors=None):
    if colors is None:
        colors = ['C', 'lightcoral']

    ax = plt.subplot()
    assert len(costs) == len(labels)
    for i in range(len(costs)):
        ax.plot(costs[i], color=colors[i], label=labels[i])
    set_ax_gray(ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('num epochs')
    ax.set_ylabel('cost')
    
    plt.rcParams['figure.dpi'] = 1000 #分辨率
    plt.show()

#----------------------------------------


class BpNN(object):
    def __init__(self, layer_dims_, learning_rate=0.1, seed=16, initializer='he', optimizer='gd'):

        self.layer_dims_ = layer_dims_
        self.learning_rate = learning_rate
        self.seed = seed
        self.initializer = initializer
        self.optimizer = optimizer

    def fit(self, X_, y_, num_epochs=100):
        m, n = X_.shape
        layer_dims_ = copy.deepcopy(self.layer_dims_)
        layer_dims_.insert(0, n)

        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)

        assert self.initializer in ('he', 'xavier')

        if self.initializer == 'he':
            self.parameters_ = xavier_initializer(layer_dims_, self.seed)
        elif self.initializer == 'xavier':
            self.parameters_ = xavier_initializer(layer_dims_, self.seed)

        assert self.optimizer in ('gd', 'sgd', 'adam', 'momentum')
        if self.optimizer == 'gd':
            parameters_, costs = self.optimizer_gd(X_, y_, self.parameters_, num_epochs, self.learning_rate)
        elif self.optimizer == 'sgd':
            parameters_, costs = self.optimizer_sgd(X_, y_, self.parameters_, num_epochs, self.learning_rate, self.seed)
        elif self.optimizer == 'momentum':
            parameters_, costs = self.optimizer_sgd_monment(X_, y_, self.parameters_, beta=0.9, num_epochs=num_epochs,
                                                            learning_rate=self.learning_rate, seed=self.seed)
        elif self.optimizer == 'adam':
            parameters_, costs = self.optimizer_sgd_adam(X_, y_, self.parameters_, beta1=0.9, beta2=0.999, epsilon=1e-7,
                                                         num_epochs=num_epochs, learning_rate=self.learning_rate,
                                                         seed=self.seed)

        self.parameters_ = parameters_
        self.costs = costs

        return self

    def predict(self, X_):
        if not hasattr(self, "parameters_"):
            raise Exception('you have to fit first before predict.')

        a_last, _ = self.forward_L_layer(X_, self.parameters_)
        if a_last.shape[1] == 1:
            predict_ = np.zeros(a_last.shape)
            predict_[a_last>=0.5] = 1
        else:
            predict_ = np.argmax(a_last, axis=1)
        return predict_

    def compute_cost(self, y_hat_, y_):
        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)
        if y_.shape[1] == 1:
            cost = cross_entry_sigmoid(y_hat_, y_)
        else:
            cost = cross_entry_softmax(y_hat_, y_)
        return cost

    def backward_one_layer(self, da_, cache_, activation_):
        # 在activation_ 为'softmax'时， da_实际上输入是y_， 并不是
        (a_pre_, w_, b_, z_) = cache_
        m = da_.shape[0]

        assert activation_ in ('sigmoid', 'relu', 'softmax')

        if activation_ == 'sigmoid':
            dz_ = sigmoid_backward(da_, z_)
        elif activation_ == 'relu':
            dz_ = relu_backward(da_, z_)
        else:
            dz_ = softmax_backward(da_, z_)

        dw = np.dot(dz_.T, a_pre_) / m
        db = np.sum(dz_, axis=0, keepdims=True) / m
        da_pre = np.dot(dz_, w_)

        assert dw.shape == w_.shape
        assert db.shape == b_.shape
        assert da_pre.shape == a_pre_.shape

        return da_pre, dw, db

    def backward_L_layer(self, a_last, y_, caches):

        grads = {}
        L = len(caches)

        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)

        if y_.shape[1] == 1:  # 目标值只有一列表示为二分类
            da_last = -(y_ / a_last - (1 - y_) / (1 - a_last))
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(da_last, caches[L - 1], 'sigmoid')

        else:  # 经过one hot，表示为多分类

            # 在计算softmax的梯度时，可以直接用 dz = a - y可计算出交叉熵损失函数对z的偏导， 所以这里第一个参数输入直接为y_
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(y_, caches[L - 1], 'softmax')

        grads['da' + str(L)] = da_pre_L_1
        grads['dW' + str(L)] = dwL_
        grads['db' + str(L)] = dbL_

        for i in range(L - 1, 0, -1):
            da_pre_, dw, db = self.backward_one_layer(grads['da' + str(i + 1)], caches[i - 1], 'relu')

            grads['da' + str(i)] = da_pre_
            grads['dW' + str(i)] = dw
            grads['db' + str(i)] = db

        return grads

    def forward_one_layer(self, a_pre_, w_, b_, activation_):
        z_ = np.dot(a_pre_, w_.T) + b_
        assert activation_ in ('sigmoid', 'relu', 'softmax')

        if activation_ == 'sigmoid':
            a_ = sigmoid(z_)
        elif activation_ == 'relu':
            a_ = relu(z_)
        else:
            a_ = softmax(z_)

        cache_ = (a_pre_, w_, b_, z_)  # 将向前传播过程中产生的数据保存下来，在向后传播过程计算梯度的时候要用上的。
        return a_, cache_

    def forward_L_layer(self, X_, parameters_):
        L_ = int(len(parameters_) / 2)
        caches = []
        a_ = X_
        for i in range(1, L_):
            w_ = parameters_['W' + str(i)]
            b_ = parameters_['b' + str(i)]
            a_pre_ = a_
            a_, cache_ = self.forward_one_layer(a_pre_, w_, b_, 'relu')
            caches.append(cache_)

        w_last = parameters_['W' + str(L_)]
        b_last = parameters_['b' + str(L_)]

        if w_last.shape[0] == 1:
            a_last, cache_ = self.forward_one_layer(a_, w_last, b_last, 'sigmoid')
        else:
            a_last, cache_ = self.forward_one_layer(a_, w_last, b_last, 'softmax')

        caches.append(cache_)
        return a_last, caches

    def optimizer_gd(self, X_, y_, parameters_, num_epochs, learning_rate):
        costs = []
        for i in range(num_epochs):
            a_last, caches = self.forward_L_layer(X_, parameters_)
            grads = self.backward_L_layer(a_last, y_, caches)

            parameters_ = update_parameters_with_gd(parameters_, grads, learning_rate)
            cost = self.compute_cost(a_last, y_)

            costs.append(cost)

        return parameters_, costs

    def optimizer_sgd(self, X_, y_, parameters_, num_epochs, learning_rate, seed):
        '''
        sgd中，更新参数步骤和gd是一致的，只不过在计算梯度的时候是用一个样本而已。
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]
        for _ in range(num_epochs):
            random_index = np.random.randint(0, m_)

            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            parameters_ = update_parameters_with_sgd(parameters_, grads, learning_rate)

            a_last_cost, _ = self.forward_L_layer(X_, parameters_)

            cost = self.compute_cost(a_last_cost, y_)

            costs.append(cost)

        return parameters_, costs

    def optimizer_sgd_monment(self, X_, y_, parameters_, beta, num_epochs, learning_rate, seed):
        '''

        :param X_:
        :param y_:
        :param parameters_: 初始化的参数
        :param v_:          梯度的指数加权移动平均数
        :param beta:        冲量大小，
        :param num_epochs:
        :param learning_rate:
        :param seed:
        :return:
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]
        velcoity = initialize_velcoity(parameters_)
        for _ in range(num_epochs):
            random_index = np.random.randint(0, m_)

            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            parameters_, v_ = update_parameters_with_sgd_momentum(parameters_, grads, velcoity, beta,
                                                                           learning_rate)
            a_last_cost, _ = self.forward_L_layer(X_, parameters_)
            cost = self.compute_cost(a_last_cost, y_)
            costs.append(cost)

        return parameters_, costs

    def optimizer_sgd_adam(self, X_, y_, parameters_, beta1, beta2, epsilon, num_epochs, learning_rate, seed):
        '''

        :param X_:
        :param y_:
        :param parameters_: 初始化的参数
        :param v_:          梯度的指数加权移动平均数
        :param beta:        冲量大小，
        :param num_epochs:
        :param learning_rate:
        :param seed:
        :return:
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]
        velcoity, square_grad = initialize_adam(parameters_)
        for epoch in range(num_epochs):
            random_index = np.random.randint(0, m_)

            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            parameters_, velcoity, square_grad = update_parameters_with_sgd_adam(parameters_, grads, velcoity,
                                                                                          square_grad, epoch + 1,
                                                                                          learning_rate, beta1, beta2,
                                                                                          epsilon)
            a_last_cost, _ = self.forward_L_layer(X_, parameters_)
            cost = self.compute_cost(a_last_cost, y_)
            costs.append(cost)

        return parameters_, costs


if __name__ == '__main__':
    # 5.5

    # data_path = r'C:\Users\Administrator\Desktop\Data\watermelon3_0_Ch.csv'
    # data3 = pd.read_csv(data_path, index_col=0)
    # data = pd.get_dummies(data3, columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
    # data['好瓜'].replace(['是', '否'], [1, 0], inplace=True)
    # X_test = data.drop('好瓜', axis=1)
    # y_test = data['好瓜']
    
    # bp = BpNN([3, 1], learning_rate=0.1, optimizer='gd')
    # bp.fit(X_test.values, y_test.values, num_epochs=1000)   #迭代次数200、500、1000测试

    # bp1 = BpNN([3, 1], learning_rate=0.1, optimizer='sgd')
    # bp1.fit(X_test.values, y_test.values, num_epochs=1000)
    
    # plot_costs([bp.costs, bp1.costs], ['gd_cost', 'sgd_cost'])



    #5.6
    
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    X = (X - np.mean(X, axis=0)) / np.var(X, axis=0)

    y = pd.Series(iris['target_names'][iris['target']])
    y = pd.get_dummies(y)

    bp = BpNN([3, 3], learning_rate=0.003, optimizer='adam')
    bp.fit(X.values, y.values, num_epochs=2000)

    bp1 = BpNN([3, 3], learning_rate=0.003, optimizer='gd')
    bp1.fit(X.values, y.values, num_epochs=2000)

    plot_costs([bp.costs, bp1.costs], ['adam_cost', 'gd_cost'])
    
    