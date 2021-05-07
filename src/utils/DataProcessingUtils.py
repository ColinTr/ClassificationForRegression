"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

from sklearn.preprocessing import PowerTransformer
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random


def one_hot_encode():
    # TODO
    return None


def box_cox(Y_train, Y_test):
    """
    Fits a box-cox transformation on the training goal values and applies it on the training and testing goal values.
    :param Y_train: list or np.array : The training goal values to transform
    :param Y_test: list or np.array :  The testing goal values to transform
    :return: Y_train, Y_test : The transformed Y_train and Y_test
    """
    # Fit BoxCox on the training dataset
    transform = PowerTransformer(method='box-cox')  # Only works with strictly positive values !
    transform.fit(Y_train.reshape(-1, 1))

    # Apply BoxCox on the training AND testing datasets
    # sns.displot(Y)
    # plt.savefig("Y_orignal.png")
    Y_train = transform.transform(Y_train.reshape(-1, 1))
    Y_train = np.concatenate(Y_train).ravel()
    Y_test = transform.transform(Y_test.reshape(-1, 1))
    Y_test = np.concatenate(Y_test).ravel()
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
