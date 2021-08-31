"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from sklearn.isotonic import IsotonicRegression
import numpy as np


class CDFRegressor():
    def __init__(self):
        super(CDFRegressor, self).__init__()
        self.thresholds_real_values = None
        self.n_thresholds = None
        self.values_between_thresholds = None

    def fit(self, thresholds_real_values):
        self.thresholds_real_values = thresholds_real_values
        self.n_thresholds = len(thresholds_real_values)

        self.values_between_thresholds = []
        for tmp_index in range(len(thresholds_real_values) - 1):
            self.values_between_thresholds.append((thresholds_real_values[tmp_index] + thresholds_real_values[tmp_index + 1]) / 2.0)

    def get_thresholds_columns(self, X):
        return [e for e in X.columns if 'threshold' in e.split('_') and '1|X)' in e.split('_')]

    def predict(self, X):
        predictions = []

        for classifiers_prediction in np.array(X[self.get_thresholds_columns(X)]):
            isotonic_regression_model = IsotonicRegression()
            isotonic_regression_model.fit(self.thresholds_real_values, classifiers_prediction)
            isotonic_prediction = isotonic_regression_model.predict(self.thresholds_real_values)

            differences = []
            for tmp_index in range(self.n_thresholds - 1):
                differences.append(isotonic_prediction[tmp_index + 1] - isotonic_prediction[tmp_index])

            weighted_mean = 0
            for yi, Pi in zip(self.values_between_thresholds, differences):
                weighted_mean += Pi * yi

            predictions.append(weighted_mean)

        return predictions

    def predict_largest_diff(self, X):
        predictions = []

        for classifiers_prediction in np.array(X[self.get_thresholds_columns(X)]):
            # No isotonic regression here
            differences = []
            for tmp_index in range(self.n_thresholds - 1):
                differences.append(classifiers_prediction[tmp_index + 1] - classifiers_prediction[tmp_index])

            predictions.append(self.values_between_thresholds[np.argmax(differences)])

        return predictions

    def predict_largest_diff_with_isotonic_regression(self, X):
        predictions = []

        for classifiers_prediction in np.array(X[self.get_thresholds_columns(X)]):
            isotonic_regression_model = IsotonicRegression()
            isotonic_regression_model.fit(self.thresholds_real_values, classifiers_prediction)
            isotonic_prediction = isotonic_regression_model.predict(self.thresholds_real_values)

            differences = []
            for tmp_index in range(self.n_thresholds - 1):
                differences.append(isotonic_prediction[tmp_index + 1] - isotonic_prediction[tmp_index])

            predictions.append(self.values_between_thresholds[np.argmax(differences)])

        return predictions

    def fit_predict(self, thresholds_real_values, X):
        self.fit(thresholds_real_values)
        return self.predict(X)

    def fit_predict_largest_diff(self, thresholds_real_values, X):
        self.fit(thresholds_real_values)
        return self.predict_largest_diff(X)