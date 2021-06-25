"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from . import BaseModel
import pandas as pd
import numpy as np
import platform
import os

if platform.system() == "Windows":
    if os.environ.get('KhiopsHome') is not None:
        from pykhiops.sklearn import KhiopsClassifier
else:
    if os.path.exists(os.path.join(os.environ["HOME"], "pykhiops", "lib")):
        from pykhiops.sklearn import KhiopsClassifier


class PyKhiopsC(BaseModel.BaseModel):
    def __init__(self):
        super(PyKhiopsC, self).__init__()
        self.model = KhiopsClassifier()
        self.classes = None

    def fit(self, X_train, Y_train):
        self.model.fit(X=X_train, y=Y_train)
        self.classes = list(np.unique(Y_train))

    def extract_features(self, X, Y):
        # model_score = acc(self.model.predict(X=X), Y)
        extracted_features = pd.DataFrame([])

        # Extract the conditional probabilities of each class
        predicted_class_probabilities = np.array(self.model.predict_proba(X))
        for class_index, index in zip(self.classes, range(len(self.classes))):
            extracted_features["P(C_" + str(class_index) + "|X)"] = predicted_class_probabilities[:, index]

        # TODO : Extract more features

        return extracted_features, None
