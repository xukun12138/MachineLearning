# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:52:18 2020

@author: Administrator
"""

import numpy as np

class Classifier(object):
    def __init__(self, attr_count, learn_rate=0.05):
        self.__attr_count__ = attr_count
        self.__learn_rate__ = learn_rate
        self.__weight__ = np.zeros(shape=[attr_count + 1], dtype=np.float32)

    def fit(self, value, label):
        value = np.append(value, [1.0])
        linear_result = np.dot(value, self.__weight__)
        sigmoid_result = 1.0 / (np.exp(-linear_result) + 1.0)
        for idx in range(self.__attr_count__ + 1):
            update_val = (sigmoid_result - label) * value[idx]
            self.__weight__[idx] -= self.__learn_rate__ * update_val

    def classify(self, value):
        value = np.append(value, [1.0])
        linear_result = np.dot(value, self.__weight__)
        if (1.0 / (np.exp(-linear_result) + 1.0)) > 0.5:
            return 1
        else:
            return 0

    
def test_main(value, label, attr_count):
    TRAIN_TIMES = 1
    batch_size = len(value) // 10
    total_correct_times = 0
    for idx in range(10):
        #print('10折交叉验证：当前第 %d 次' % (idx + 1))
        correct_times = 0
        classifier = Classifier(attr_count)
        value_train = np.append(value[0:idx * batch_size], value[(idx + 1) * batch_size:], axis=0)
        label_train = np.append(label[0:idx * batch_size], label[(idx + 1) * batch_size:], axis=0)
        value_test = value[idx * batch_size:(idx + 1) * batch_size]
        label_test = label[idx * batch_size:(idx + 1) * batch_size]
        for repeat in range(TRAIN_TIMES):
            for sub_idx in range(len(value_train)):
                classifier.fit(value_train[sub_idx], label_train[sub_idx])
        for sub_idx in range(len(value_test)):
            result = classifier.classify(value_test[sub_idx])
            if result != label_test[sub_idx]:
                correct_times += 1
        total_correct_times += correct_times
        #print('准确率：%.2f%%\n' % (correct_times * 100 / len(value_test)))
    print('10折交叉验证平均错误率：'+str(total_correct_times / len(value)))
    total_times = len(value)
    correct_times = 0
    for idx in range(total_times):
        #print('留一法第 %d 次，共 %d 次' % (idx, total_times))
        classifier = Classifier(attr_count)
        value_train = np.append(value[0:idx], value[(idx + 1):], axis=0)
        label_train = np.append(label[0:idx], label[(idx + 1):], axis=0)
        value_test = value[idx]
        label_test = label[idx]
        for repeat in range(TRAIN_TIMES):
            for sub_idx in range(len(value_train)):
                classifier.fit(value_train[sub_idx], label_train[sub_idx])
        result = classifier.classify(value_test)
        if result != label_test:
            correct_times += 1
    print('留一法错误率：'+ str((correct_times / total_times)))
    
if __name__ == '__main__':
    #print('Wine数据集')
    # 读入并打乱 Wine 数据集
    data_wine = open('C:\\Users\\Administrator\\Desktop\\Data\\wine.data').readlines()
    #print(data_wine)
    np.random.shuffle(data_wine)
    # 使用数组切片操作，分离数据和标签
    wine_value = np.ndarray([len(data_wine), 13], np.float32)
    #print(wine_value)
    wine_label = np.ndarray([len(data_wine)], np.int32)
    for outer_idx in range(len(data_wine)):
        data = data_wine[outer_idx].strip('\n').split(',')
        wine_value[outer_idx] = data[1:]
        wine_label[outer_idx] = data[0]
    # 进行训练和测试
    test_main(wine_value, wine_label, 13)
#-------------------------------------------------   
    #print('Iris数据集')
    # 读入并打乱 Iris 数据集
    data_iris = open('C:\\Users\\Administrator\\Desktop\\Data\\iris.data').readlines()
    #print(type(data_iris))
    np.random.shuffle(data_iris)
    # 使用数组切片操作，分离数据和标签
    iris_value = np.ndarray([len(data_iris),4], np.float32)
    #print(iris_value)
    
    '''
    iris_label = np.ndarray([len(data_iris)], np.int32)
    for outer_idx in range(len(data_iris)):
        data = data_iris[outer_idx].strip('\n').split(',')
        iris_value[outer_idx] = data[1:]
        iris_label[outer_idx] = data[0]
    #进行训练和测试
    test_main(iris_value, iris_label, 4)
    '''