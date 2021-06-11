"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import os
import platform
import sys

# Check the Khiops python directory on linux
if platform.system() == "Linux":
    khiopsPythonDir = os.path.join(os.environ["HOME"], "pykhiops")
    if not os.path.isdir(khiopsPythonDir):
        raise ValueError("The pykhiops library path is not correctly set. Copy the library to " + khiopsPythonDir)

# Specify Khiops python lib path
khiopsLibPath = os.path.join(os.environ["KhiopsHome"], "python", "lib") if platform.system() == "Windows" else os.path.join(os.environ["HOME"], "pykhiops", "lib")
if not khiopsLibPath in sys.path:
    sys.path.append(khiopsLibPath)

from pykhiops import PyKhiopsClassifier, acc
from . import BaseModel
import pandas as pd


class PyKhiopsC(BaseModel.BaseModel):
    def __init__(self):
        super(PyKhiopsC, self).__init__()
        self.model = PyKhiopsClassifier()

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def extract_features(self, X, Y):
        model_score = acc(self.model.predict(X=X), Y)
        extracted_features = pd.DataFrame([])

        # Extract the conditional probabilities of each class
        proba_dataframe = self.model.predict_proba(X)
        # Since Khiops gives a dataframe with a first column being the predicted class and the others columns,
        #     we drop the first column
        predicted_class_probabilities = proba_dataframe.drop(proba_dataframe.columns[0], axis=1).to_numpy()
        for class_index, index in zip(self.model.classes_, range(len(self.model.classes_))):
            extracted_features["P(C_" + str(class_index) + "|X)"] = predicted_class_probabilities[:, index]

        # TODO : Extract more features

        return extracted_features, model_score
