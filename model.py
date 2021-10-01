""" Created by MrBBS """
# 8/12/2021
# -*-encoding:utf-8-*-

from pyvi import ViTokenizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np

class FeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, *_):
        return self

    def transform(self, X, y=None, **fit_params):
        result = [ViTokenizer.tokenize(text.strip().lower()) for text in X]
        return np.array(result)


class SVMModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf-svm", SGDClassifier(loss='log', alpha=1e-3))
        ])

        return pipe_line
