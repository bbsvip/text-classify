""" Created by MrBBS """
# 8/12/2021
# -*-encoding:utf-8-*-
from model import SVMModel, FeatureTransformer
import pickle
import numpy as np

class TextClassify():
    def __init__(self):
        self.model = pickle.load(open('svm.sav', 'rb'))

    def classify(self, text: str):
        """
        Phân loại câu

        :param text: Câu cần phân loại
        :return: (str) Loại nội dung của câu
        """
        predicted = self.model.predict([text.lower()])
        return predicted[0]
