"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd


def compute_all_metrics(y_true, y_pred, n, p):
    """
    Computes the following metrics : MAE, MSE, RMSE, R² and Adjusted R².
    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated target values.
    :param n: Number of instances.
    :param p: Number of attributes of the dataset.
    :return: A dict containing all the losses.
    """
    return {"mean_absolute_error": compute_mean_absolute_error(y_true, y_pred),
            "mean_squared_error": compute_mean_squared_error(y_true, y_pred),
            "root_mean_squared_error": compute_root_mean_squared_error(y_true, y_pred),
            "r_squared": compute_r_squared(y_true, y_pred),
            "adjusted_r_squared": compute_adjusted_r_squared(y_true, y_pred, n, p)}


def compute_mean_absolute_error(y_true, y_pred):
    """
    Mean absolute error regression loss.
    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated target values.
    :return: loss, a float.
    """
    return mean_absolute_error(y_true, y_pred)


def compute_mean_squared_error(y_true, y_pred):
    """
    Mean squared error regression loss.
    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated target values.
    :return: loss, a float.
    """
    return mean_squared_error(y_true, y_pred)


def compute_root_mean_squared_error(y_true, y_pred):
    """
    Root mean squared error regression loss.
    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated target values.
    :return: loss, a float.
    """
    return mean_squared_error(y_true, y_pred, squared=False)


def compute_r_squared(y_true, y_pred):
    """
    R-squared regression score function.
    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated target values.
    :return: loss, a float.
    """
    return r2_score(y_true, y_pred)


def compute_adjusted_r_squared(y_true, y_pred, n, p):
    """
    Adjusted R-squared regression score function.
    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated target values.
    :param n: Number of instances.
    :param p: Number of attributes of the dataset.
    :return: loss, a float.
    """
    return 1 - (1 - compute_r_squared(y_true, y_pred)) * ((n - 1) / (1 - p - 1))
