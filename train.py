""" Created by MrBBS """
# 8/12/2021
# -*-encoding:utf-8-*-
from model import SVMModel
import pickle
import numpy as np

class TextClassificationPredict(object):
    def __init__(self):
        self.test = None

    def get_train_data(self):
        # Tạo train data
        train_x = []
        train_y = []
        data = open('data.txt', 'r', encoding='utf-8').readlines()
        np.random.shuffle(data)
        for line in data:
            try:
                label, doc = line.strip().split('\t')
                train_x.append(doc.strip())
                train_y.append(label.strip())
            except ValueError:
                pass

        # init model
        model1 = SVMModel()
        print('train svm')
        clf1 = model1.clf.fit(train_x, train_y)
        print('save SVM')
        pickle.dump(clf1, open('svm.sav', 'wb'))

        # Test model
        predicted1 = clf1.predict(["185 Lê Trọng Tấn"])
        print(predicted1)
        print(clf1.predict_proba(["185 Lê Trọng Tấn"]))


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.get_train_data()
