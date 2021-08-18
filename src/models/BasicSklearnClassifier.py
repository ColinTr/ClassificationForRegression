"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from importlib import import_module
from . import BaseModel
import pandas as pd
import numpy as np


class BasicSklearnClassifier(BaseModel.BaseModel):
    def __init__(self, model_name):
        super(BasicSklearnClassifier, self).__init__()

        module_path, class_name = model_name.rsplit('.', 1)

        module = import_module(module_path)

        self.model = getattr(module, class_name)()
        self.classes = None

    def fit(self, X_train, Y_train):
        self.model.fit(np.array(X_train), np.array(Y_train))
        self.classes = list(np.unique(Y_train))

    def extract_features(self, X, Y):
        extracted_features = pd.DataFrame([])

        # Extract the conditional probabilities of each class
        predicted_class_probabilities = self.model.predict_proba(np.ascontiguousarray(X))
        for class_index, index in zip(self.classes, range(len(self.classes))):
            extracted_features["P(C_" + str(class_index) + "|X)"] = predicted_class_probabilities[:, index]

        # TODO : Extract more features

        return extracted_features
