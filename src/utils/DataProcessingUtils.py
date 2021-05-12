"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random


def one_hot_encode():
    # (optional) TODO
    return None


def box_cox(Y_train, Y_test):
    """
    Fits a box-cox transformation on the training goal values and applies it on the training and testing goal values.
    :param Y_train: list or np.array : The training goal values to transform
    :param Y_test: list or np.array :  The testing goal values to transform
    :return: Y_train, Y_test : The transformed Y_train and Y_test
    """
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    # Since BoxCox can only be applied to positive data, we first scale the data to [1, 2]
    scaler = MinMaxScaler(feature_range=(1, 2))
    scaler.fit(Y_train)
    Y_train = scaler.transform(Y_train)
    Y_test = scaler.transform(Y_test)  # Risk of still having negative data in the test goal variable...

    # Fit BoxCox on the training dataset
    transform = PowerTransformer(method='box-cox')  # Only works with strictly positive values !
    transform.fit(Y_train)

    # Apply BoxCox on the training AND testing datasets
    # sns.displot(Y)
    # plt.savefig("Y_orignal.png")
    Y_train = transform.transform(Y_train)
    Y_test = transform.transform(Y_test)
    # sns.displot(Y_train)
    # plt.savefig("Y_box-cox_transformed.png")
    return Y_train, Y_test


def normalize(X_train, X_test):
    """
    Fits a StandardScaler on the training data and applies it on the training and testing data.
    :param X_train: np.array or pd.DataFrame : The training data
    :param X_test: np.array or pd.DataFrame : The testing data
    :return: X_train, X_test : The transformed X_train and X_test
    """
    # Fit normalization on the training dataset
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    # Apply normalization on the training AND testing datasets
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    return X_train, X_test


def kfold_train_test_split(n_indexes, k_folds, seed=None):
    """
    Generate indexes and randomly puts them in k_folds lists.
    :param n_indexes: int : The number of indexes to use
    :param k_folds: int : The number of lists to use
    :param seed: int : The seed for the random shuffling
    :return: list[list[int]] : The shuffled indexes inside their k_folds lists
    """
    random.seed(seed)

    indexes_list = list(range(0, n_indexes))

    random.shuffle(indexes_list)

    return np.array_split(indexes_list, k_folds)


def detect_class_columns(header):
    """
    Detect from a header which columns have a string "class"
    :param header: The header
    :return: A List of the indexes of the columns with the string "class" inside
    """
    col_index = 0
    regression_goal_var_index = -1
    class_cols_indexes = []
    for column_name in header:
        if column_name.split('_')[0] == 'class':
            class_cols_indexes.append(col_index)
        elif column_name == 'reg_goal_var':
            regression_goal_var_index = col_index
        col_index = col_index + 1
    return class_cols_indexes, regression_goal_var_index
