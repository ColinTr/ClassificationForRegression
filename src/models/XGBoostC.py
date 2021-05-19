"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from xgboost import XGBClassifier
from . import BaseModel
import pandas as pd
import numpy as np


class XGBoostC(BaseModel.BaseModel):
    def __init__(self):
        super(XGBoostC, self).__init__()
        self.model = XGBClassifier(n_estimators=100, use_label_encoder=False, objective='binary:logistic', eval_metric='logloss')

    def fit(self, X_train, Y_train):
        self.model.fit(np.ascontiguousarray(X_train), Y_train)

    def extract_features(self, X, Y):
        model_score = self.model.score(np.ascontiguousarray(X), Y)
        extracted_features = pd.DataFrame([])

        # Extract the conditional probabilities of each class
        predicted_class_probabilities = self.model.predict_proba(X)
        for index in range(0, predicted_class_probabilities.shape[1]):
            extracted_features["P(C_" + str(index) + "|X)"] = predicted_class_probabilities[:, index]

        # TODO : Extract more features

        return extracted_features, model_score