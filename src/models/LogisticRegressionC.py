"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from sklearn.linear_model import LogisticRegression
from . import BaseModel
import pandas as pd


class LogisticRegressionC(BaseModel.BaseModel):
    def __init__(self):
        super(LogisticRegressionC, self).__init__()
        self.model = LogisticRegression(n_jobs=-1)

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def extract_features(self, X, Y):
        model_score = self.model.score(X, Y)
        extracted_features = pd.DataFrame([])

        # Extract the conditional probabilities of each class
        predicted_class_probabilities = self.model.predict_proba(X)
        for index in range(0, predicted_class_probabilities.shape[1]):
            extracted_features["P(C_" + str(index) + "|X)"] = predicted_class_probabilities[:, index]

        # TODO : Extract more features

        return extracted_features, model_score
