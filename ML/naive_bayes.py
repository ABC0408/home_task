# -*- coding:utf-8 -*-
import arff
import warnings
import numpy as np
import pandas as pd
from math import log
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
np.set_printoptions(linewidth=800)


def naive_bayes(x, y, predict):
    n_class = []
    for i in y:
        if i not in n_class:
            n_class.append(i)
    label_num = len(n_class)
    train_num, dim = x.shape
    p = [0] * label_num
    for (i, label) in enumerate(n_class):
        p_c = len(y[y == label]) / train_num
        for (j, x_i) in enumerate(predict):
            x_current = x[y == label]
            count_num = 1e-10
            for t in x_current[:, j]:
                if t == x_i:
                    count_num += 1
            p[i] += log(count_num / len(x_current))
        p[i] += log(p_c)
    max_index = np.argmax(p)
    return n_class[int(max_index)]


def load_data(name, train_size):

    data = arff.load(open('%s.arff' % name, 'r'))
    data = np.array(pd.DataFrame(data['data']).dropna())
    label_encoder = LabelEncoder()
    encoded = np.zeros(data.shape)
    for i in range(len(data[0])):
        encoded[:, i] = label_encoder.fit_transform(data[:, i])
    x, y = encoded[:, :-1], encoded[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=train_size)
    return x_train, x_test, y_train, y_test


def test(dataset, train_size):

    x_train, x_test, y_train, y_test = load_data(dataset, train_size)
    output = []
    for i in x_test:
        output.append(naive_bayes(x_train, y_train, i))
    accuracy = accuracy_score(y_test, output) * 100
    print('\n%s test accuracy: %.2f%%' % (dataset, accuracy),
          '\nTrue:\n', y_test, '\nPredict：\n', np.array(output))


if __name__ == '__main__':

    # first parameter： dataset name
    # second parameter: train size
    test('weather', 0.7)
    test('soybean', 0.96)

'''
weather test accuracy: 80.00% 
True:
 [ 1.  0.  1.  1.  1.] 
Predict：
 [ 1.  0.  1.  1.  0.]

soybean test accuracy: 95.65% 
True:
 [  5.   0.   4.   9.   4.   9.  10.   1.   8.  13.  14.   0.   1.   9.   0.   3.   9.   4.   0.   9.   5.   4.   3.] 
Predict：
 [  5.   0.   4.   9.   4.   9.   4.   1.   8.  13.  14.   0.   1.   9.   0.   3.   9.   4.   0.   9.   5.   4.   3.]

'''

