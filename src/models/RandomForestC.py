"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from sklearn.ensemble import RandomForestClassifier
from . import BaseModel
import pandas as pd
import logging


class RandomForestC(BaseModel.BaseModel):
    def __init__(self):
        super(RandomForestC, self).__init__()
        self.model = RandomForestClassifier()

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def extract_features(self, X_test, Y_test):
        model_score = self.model.score(X_test, Y_test)
        extracted_features = pd.DataFrame([])

        conditional_probability_of_the_class = []
        predicted_class_probabilities = self.model.predict_proba(X_test)
        for index in range(0, len(predicted_class_probabilities)):
            conditional_probability_of_the_class.append(predicted_class_probabilities[index][Y_test[index]])

        extracted_features["P(C|X)"] = conditional_probability_of_the_class
        # TODO : Extract more features

        return extracted_features, model_score
