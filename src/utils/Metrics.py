"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
import numpy as np
import logging


def compute_all_metrics(y_true, y_pred, n, p):
    """
    Computes the following metrics : MAE, MSE, RMSE, R² and Adjusted R².
    NOTE : The order of y_true and y_pred is important, do not exchange.
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


def compute_log_losses(df):
    """
    For each class_X column inside the given dataframe of predictions, compute the mean log loss.
    :param df: The dataframe of predictions. Header must be a list of : class_ + 'class_number'
    :return: The mean log loss of every class.
    """
    classes_mean_log_loss_dict = {}

    classes_names = []
    for column_name in list(df.columns.values):
        if 'class' in column_name.split('_') and 'predicted' not in column_name.split('_'):
            classes_names.append(column_name)

    for class_name in classes_names:
        # Log loss is undefined for p=0 or p=1, so probabilities are clipped to max(eps, min(1 - eps, p))
        eps = 1e-15
        log_loss = (-1 / len(df[class_name])) * np.sum(np.log(np.maximum(eps, np.minimum(1 - eps, np.array(df[class_name])))))

        if len(classes_names) > 1:
            tmp_class_name = 'class_' + class_name.split('_')[1]
        else:
            tmp_class_name = 'class_0'
        classes_mean_log_loss_dict[tmp_class_name + '_mean_log_loss'] = [log_loss]

    # Either return individual losses with
    # return pd.DataFrame(classes_mean_log_loss_dict)

    # Or return the mean log loss of all the classifiers with
    if len(classes_mean_log_loss_dict.values()) > 0:
        return np.mean([values[0] for values in classes_mean_log_loss_dict.values()])
    else:
        return None


def compute_mean_roc_auc_score(df):
    """
    For each threshold (so classifier), compute the mean roc auc score of all of the classes.
    If two classes are found for each threshold, we are in a binary classification configuration, otherwise multiclass.
    :param df: The dataframe of predictions. Header must be a list of : threshold_' + thr_number + '_P(C_1|X)
        and of 'class' + cla_number (or just 'class' in multiclass configuration).
    :return: The mean ROC AUC score of all of the classifiers.
    """
    # For every threshold, we have certain number of possible classes
    # If it is greater than 2, we are in a multiclass configuration
    thresholds_columns_dict = {}
    classes = []

    # We start by sorting all the classes columns by threshold (classifier)
    for column_name in list(df.columns.values):
        if 'threshold' in column_name.split('_'):
            threshold_number = column_name.split('_')[1]
            class_number = int(column_name.split('_')[3].split('|')[0])
            classes.append(class_number)
            if threshold_number in thresholds_columns_dict.keys():
                thresholds_columns_dict[threshold_number].append(column_name)
            else:
                thresholds_columns_dict[threshold_number] = [column_name]

    if len(thresholds_columns_dict.values()) == 0:
        return None

    highest_number_of_classes_of_threshold_found = np.max([len(elem) for elem in thresholds_columns_dict.values()])

    if highest_number_of_classes_of_threshold_found == 2:
        # We are in a binary classification configuration

        classifiers_roc_auc_scores = []

        for threshold_col_list in thresholds_columns_dict.values():
            # For all the classes of a given threshold :
            threshold_number = threshold_col_list[0].split('_')[1]

            tmp_class_name = 'class_' + str(threshold_number)

            y_true = list(df[tmp_class_name])

            if len(np.unique(y_true)) == 2:
                # As in the documentation, we use the 'probability of the class with the greater label' for y_pred
                y_pred_proba = list(df['threshold_' + str(threshold_number) + '_P(C_1|X)'])
                classifiers_roc_auc_scores.append(roc_auc_score(y_true, y_pred_proba))
            else:
                logging.warning("Only one class present in binary classification data sample")

        computed_roc_auc_score = np.mean(classifiers_roc_auc_scores)

    elif highest_number_of_classes_of_threshold_found > 2:
        # Then we are in a multiclass configuration
        if len(thresholds_columns_dict.keys()) > 1:
            raise ValueError('More than one classifier for a multiclass configuration has been found.')

        y_true = list(df['class_0'])
        y_pred_proba = []
        for threshold_col in thresholds_columns_dict['0']:
            y_pred_proba.append(list(df[threshold_col]))
        y_pred_proba = np.transpose(y_pred_proba)

        ovr_roc_auc_list = []
        for class_number in range(len(y_pred_proba[0])):
            # First, we compute the empirical probability of encountering the class
            class_probability = y_true.count(class_number) / len(y_true)

            if class_probability != 0:
                # We now encode y_true and y_pred_proba to compute the roc_auc_score in a One vs Rest manner
                encoded_y_true = [1 if val == class_number else 0 for val in y_true]
                encoded_y_pred_proba = [y_pred_proba[row_index][class_number] for row_index in range(len(y_pred_proba))]
                ovr_roc_auc_list.append((class_probability, roc_auc_score(encoded_y_true, encoded_y_pred_proba)))

        # Now we compute the weighted mean
        tmp_roc_auc_score = np.sum([ovr_roc_auc_list[row_index][0]*ovr_roc_auc_list[row_index][1] for row_index in range(len(ovr_roc_auc_list))])
        # tmp_roc_auc_score = np.mean([ovr_roc_auc_list[row_index][1] for row_index in range(len(ovr_roc_auc_list))])  # => THIS IS WHAT SKLEARN SEEMS TO BE DOING

        if len(np.unique(y_true)) == len(y_pred_proba[0]):
            computed_roc_auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            # custom_roc_auc_score_one_vs_rest(y_true, y_pred_proba)
            # print("sklearn = " + str(computed_roc_auc_score) + " & tmp = " + str(tmp_roc_auc_score))
        else:
            computed_roc_auc_score = tmp_roc_auc_score
            # print("sklearn = Error & tmp = " + str(tmp_roc_auc_score))

    else:
        raise ValueError('Incoherent number of classes for a threshold found.')

    return computed_roc_auc_score


def compute_f1_score(df, threshold=0.5):
    class_columns = [e for e in df.columns if 'class' in e.split('_') and 'predicted' not in e.split('_')]
    C0_threshold_columns = [e for e in df.columns if 'threshold' and '0|X)' in e.split('_')]

    pairs = [(e1, e2) for e1, e2 in zip(class_columns, C0_threshold_columns)]

    f1_scores = []

    for pair in pairs:
        if pair[0].split('_')[1] not in pair[1].split('_'):
            raise ValueError('Incoherent metrics folder.')

        class_column, prediction_column = np.array(df[pair[0]]), np.array(df[pair[1]])

        predicted_class_column = [0 if probability > threshold else 1 for probability in prediction_column]

        f1_scores.append(f1_score(class_column, predicted_class_column))

    return np.mean(f1_scores)
