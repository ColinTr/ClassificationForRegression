"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from sklearn.tree import DecisionTreeClassifier
from . import BaseModel
import pandas as pd
import numpy as np


class DecisionTreeC(BaseModel.BaseModel):
    def __init__(self):
        super(DecisionTreeC, self).__init__()
        self.model = DecisionTreeClassifier()

    def fit(self, X_train, Y_train):
        self.model.fit(np.ascontiguousarray(X_train), Y_train)

    def extract_features(self, X, Y):
        model_score = self.model.score(np.ascontiguousarray(X), Y)
        extracted_features = pd.DataFrame([])

        # Extract the conditional probabilities of each class
        predicted_class_probabilities = self.model.predict_proba(np.ascontiguousarray(X))
        for class_index, index in zip(self.model.classes_, range(len(self.model.classes_))):
            extracted_features["P(C_" + str(class_index) + "|X)"] = predicted_class_probabilities[:, index]

        # TODO : Extract more features

        return extracted_features, model_score
