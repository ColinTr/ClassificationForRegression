"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

from collections import OrderedDict
from os.path import isfile, join
import matplotlib.pyplot as plt
from textwrap import wrap
from os import listdir
import pandas as pd
import numpy as np
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logging_util import generate_output_path
from src.utils.logging_util import setup_logging_level


def argument_parser():
    """
    A parser to allow user to easily process any dataset using the split method and class generation he desires.
    """

    parser = argparse.ArgumentParser(usage='\n python visualisation.py [parent_folder] [output_path] [show_variance]'
                                           '[metric] [log_lvl]',
                                     description="This program allows to visualise results.")

    parser.add_argument('--parent_folder',
                        type=str,
                        help='The dataset to process',
                        required=True)

    parser.add_argument('--output_path',
                        type=str,
                        help='The folder where the figure will be saved')

    parser.add_argument('--show_variance',
                        type=str,
                        default='true',
                        choices=['true', 'false'],
                        help='Whether the variance should be shown on the graph or not')

    parser.add_argument('--metric',
                        type=str,
                        default='r_squared',
                        choices=['r_squared', 'adjusted_r_squared', 'MSE', 'RMSE', 'MAE'],
                        help='The metric to display')

    parser.add_argument('--log_lvl',
                        type=str,
                        default='info',
                        choices=["debug", "info", "warning"],
                        help='Change the log display level')

    return parser.parse_args()


def add_or_create_in_dict(dictionary, key, value):
    """
    Simple function to add a key to a dictionnary and create it if it doesn't exists.
    :param dictionary: The dictionnary in which we will add the value and create the key if needed.
    :param key: The key where the value will be added.
    :param value: The value to add.
    """
    if key not in dictionary.keys():
        dictionary[key] = value
    else:
        dictionary[key].append(value)


if __name__ == "__main__":
    args = argument_parser()

    # Setup the logging level
    setup_logging_level(args.log_lvl)

    output_path = args.output_path
    metric = args.metric

    # Format the name so we can use it to get the metrics inside the files
    if metric == 'r_squared':
        pass
    elif metric == 'adjusted_r_squared':
        pass
    elif metric == 'MSE':
        metric = 'mean_squared_error'
    elif metric == 'RMSE':
        metric = 'root_mean_squared_error'
    elif metric == 'MAE':
        metric = 'mean_absolute_error'
    else:
        raise ValueError('Unknown parameter for metric : ' + str(metric))

    # If no value was given for the 'output_path', we will generate it automatically
    if output_path is None:
        output_path = generate_output_path(args.parent_folder, ['metrics'], 'figures')
        logging.info('Generated output path : ' + output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # The files pathes will be arranged and sorted inside this dict
    sub_directories_dict = {'equal_freq': {
        'inside_bin': {},
        'below_threshold': {}
    },
        'equal_width': {
            'inside_bin': {},
            'below_threshold': {}
        }
    }

    # We start by sorting the different files found...
    # We start by exploring the content of the parent folder
    for directory in [f.path for f in os.scandir(args.parent_folder) if f.is_dir()]:
        split_path = directory.split(os.sep)  # Split the path with the system separator ('/' or '\')
        folder_path = split_path[-1].split('_')  # example : 10_bins_equal_freq_below_threshold
        bins_number = folder_path[0]

        class_generation_name = None
        steps_encoding_name = None

        if 'freq' in folder_path:
            steps_encoding_name = 'equal_freq'
        elif 'width' in folder_path:
            steps_encoding_name = 'equal_width'
        else:
            raise ValueError('Unsupported folder name in ' + str(directory))

        if 'below' in folder_path:
            class_generation_name = 'below_threshold'
        elif 'inside' in folder_path:
            class_generation_name = 'inside_bin'
        else:
            raise ValueError('Unsupported folder name in ' + str(directory))

        # Explore the first level sub-directories (the classifier level)
        for sub_path in [f.path for f in os.scandir(directory) if f.is_dir()]:
            classifier_name = sub_path.split(os.sep)[-1]

            # Explore the second level sub-directories (the regressor level)
            for sub_sub_path in [f.path for f in os.scandir(sub_path) if f.is_dir()]:
                regressor_name = sub_sub_path.split(os.sep)[-1]

                # Define the key if it doesn't exist
                if regressor_name not in sub_directories_dict[steps_encoding_name][class_generation_name].keys():
                    sub_directories_dict[steps_encoding_name][class_generation_name][regressor_name] = {}

                # Define the key if it doesn't exist
                if classifier_name not in sub_directories_dict[steps_encoding_name][class_generation_name][
                    regressor_name].keys():
                    sub_directories_dict[steps_encoding_name][class_generation_name][regressor_name][
                        classifier_name] = {}

                # Finally, add the file's path to the dictionary
                files_in_folder = [join(sub_sub_path, f) for f in listdir(sub_sub_path) if
                                   isfile(join(sub_sub_path, f))]
                if len(files_in_folder) != 1:
                    raise ValueError('More than one file in ' + str(sub_sub_path) + ' (' + files_in_folder + ')')
                sub_directories_dict[steps_encoding_name][class_generation_name][regressor_name][classifier_name][
                    bins_number] = files_in_folder[0]

    logging.debug("FINAL DICT : " + str(sub_directories_dict))

    # Once we have this dictionary, we can generate the graphs
    for steps_encoding_method in sub_directories_dict.keys():
        for class_generation_method in sub_directories_dict[steps_encoding_method].keys():
            regressor_list = sub_directories_dict[steps_encoding_method][class_generation_method]
            if len(regressor_list) > 0:
                # We will have a figure for each regression method
                for regressor_name in regressor_list.keys():
                    baseline_mean_train_metric = None
                    baseline_mean_test_metric = None
                    classifier_list = sub_directories_dict[steps_encoding_method][class_generation_method][
                        regressor_name]
                    classifier_metrics_dict = {}
                    for classifier_name in classifier_list:
                        if classifier_name == 'Standard':
                            if len(sub_directories_dict[steps_encoding_method][class_generation_method][regressor_name][
                                       classifier_name]) > 1:
                                raise ValueError('More than one Standard file in ' + str(
                                    sub_directories_dict[steps_encoding_method][class_generation_method][
                                        regressor_name][classifier_name]))
                            else:
                                baseline_file_path = list(
                                    sub_directories_dict[steps_encoding_method][class_generation_method][
                                        regressor_name][classifier_name].values())[0]
                                baseline_df = pd.read_csv(baseline_file_path)
                                baseline_mean_train_metric = np.mean(baseline_df['train_' + metric])
                                baseline_mean_test_metric = np.mean(baseline_df['test_' + metric])
                                logging.debug(
                                    'baseline mean_train_metric : {0:.4f}'.format(baseline_mean_train_metric))
                                logging.debug(
                                    'baseline mean_test_metric : {0:.4f}'.format(baseline_mean_test_metric))
                        # If it is not named 'Standard', it means that it contains metrics
                        #     for a dataset with extracted features from a classifier
                        else:
                            classifier_metrics_dict[classifier_name] = {'abscissa': [],
                                                                        'mean_of_train_' + metric: [],
                                                                        'var_of_train_' + metric: [],
                                                                        'mean_of_test_' + metric: [],
                                                                        'var_of_test_' + metric: [],
                                                                        'mean_train_log_loss': [],
                                                                        'mean_test_log_loss': [],
                                                                        'train_mean_roc_auc_score': [],
                                                                        'test_mean_roc_auc_score': []}
                            # Cast keys into ints so we can easily sort them
                            tmp_dict = {}
                            for key, value in \
                                    sub_directories_dict[steps_encoding_method][class_generation_method][
                                        regressor_name][
                                        classifier_name].items():
                                tmp_dict[int(key)] = value
                            files_in_classifier_folder = OrderedDict(sorted(tmp_dict.items(), key=lambda t: t[0]))
                            tmp_index = 0
                            for key, value in files_in_classifier_folder.items():
                                classifier_metrics_dict[classifier_name]['abscissa'].append(key)
                                tmp_df = pd.read_csv(value)
                                classifier_metrics_dict[classifier_name]['mean_of_train_' + metric].append(
                                    np.mean(tmp_df['train_' + metric]))
                                classifier_metrics_dict[classifier_name]['var_of_train_' + metric].append(
                                    np.var(tmp_df['train_' + metric]))
                                classifier_metrics_dict[classifier_name]['mean_of_test_' + metric].append(
                                    np.mean(tmp_df['test_' + metric]))
                                classifier_metrics_dict[classifier_name]['var_of_test_' + metric].append(
                                    np.var(tmp_df['test_' + metric]))
                                if 'train_mean_log_loss' in list(tmp_df.columns.values):
                                    classifier_metrics_dict[classifier_name]['mean_train_log_loss'].append(
                                        (tmp_index, np.mean(tmp_df['train_mean_log_loss'])))
                                    classifier_metrics_dict[classifier_name]['mean_test_log_loss'].append(
                                        (tmp_index, np.mean(tmp_df['test_mean_log_loss'])))
                                if 'train_mean_roc_auc_score' in list(tmp_df.columns.values):
                                    classifier_metrics_dict[classifier_name]['train_mean_roc_auc_score'].append(
                                        (tmp_index, np.mean(tmp_df['train_mean_roc_auc_score'])))
                                    classifier_metrics_dict[classifier_name]['test_mean_roc_auc_score'].append(
                                        (tmp_index, np.mean(tmp_df['test_mean_roc_auc_score'])))
                                tmp_index = tmp_index + 1

                    # We can now generate a figure with a line for each classifier
                    # ========== Test ==========
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    if baseline_mean_test_metric is not None:
                        plt.axhline(y=baseline_mean_test_metric, color='r', linestyle='--', label='baseline')
                    for key, value in classifier_metrics_dict.items():
                        mean_test_metric_df = pd.DataFrame({'x': value['abscissa'],
                                                            'y': value['mean_of_test_' + metric],
                                                            'var': value['var_of_test_' + metric]})

                        plt.plot(mean_test_metric_df["x"], mean_test_metric_df["y"], label=key, marker='o')

                        for log_loss_tuple in classifier_metrics_dict[key]['mean_test_log_loss']:
                            tag = '{0:.2f}'.format(log_loss_tuple[1])
                            for auc_tuple in classifier_metrics_dict[key]['test_mean_roc_auc_score']:
                                if auc_tuple[0] == log_loss_tuple[0]:
                                    tag += ' / {0:.2f}'.format(auc_tuple[1])
                            ax.annotate(tag, (mean_test_metric_df["x"][log_loss_tuple[0]], mean_test_metric_df["y"][log_loss_tuple[0]]))

                        if args.show_variance == 'true':
                            plt.fill_between(mean_test_metric_df["x"],
                                             mean_test_metric_df["y"] - mean_test_metric_df["var"],
                                             mean_test_metric_df["y"] + mean_test_metric_df["var"], alpha=0.2)
                    plt.legend()
                    title = ax.set_title("\n".join(wrap(
                        'Using ' + steps_encoding_method + ' & ' + class_generation_method + ', predicted with ' + regressor_name,
                        90)), fontsize=10)
                    fig.tight_layout()
                    dataset_name = os.path.basename(os.path.normpath(args.parent_folder))
                    plt.suptitle(dataset_name + ' (test)')
                    fig.subplots_adjust(top=0.88)
                    plt.savefig(os.path.join(output_path,
                                             'Test_' + metric + '_' + dataset_name + '_' + steps_encoding_method + '_' + class_generation_method + '_' + regressor_name + '.png'))
                    # ==========================

                    # ========== Train ==========
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.get_yaxis().get_major_formatter().set_useOffset(False)
                    if baseline_mean_train_metric is not None:
                        plt.axhline(y=baseline_mean_train_metric, color='r', linestyle='--', label='baseline')
                    for key, value in classifier_metrics_dict.items():
                        mean_train_metric_df = pd.DataFrame({'x': value['abscissa'],
                                                             'y': value['mean_of_train_' + metric],
                                                             'var': value['var_of_train_' + metric]})

                        plt.plot(mean_train_metric_df["x"], mean_train_metric_df["y"], label=key, marker='o')

                        for log_loss_tuple in classifier_metrics_dict[key]['mean_train_log_loss']:
                            tag = '{0:.2f}'.format(log_loss_tuple[1])
                            for auc_tuple in classifier_metrics_dict[key]['train_mean_roc_auc_score']:
                                if auc_tuple[0] == log_loss_tuple[0]:
                                    tag += ' / {0:.2f}'.format(auc_tuple[1])
                            ax.annotate(tag, (mean_train_metric_df["x"][log_loss_tuple[0]], mean_train_metric_df["y"][log_loss_tuple[0]]))

                        if args.show_variance == 'true':
                            plt.fill_between(mean_train_metric_df["x"],
                                             mean_train_metric_df["y"] - mean_train_metric_df["var"],
                                             mean_train_metric_df["y"] + mean_train_metric_df["var"], alpha=0.2)
                    plt.legend()
                    title = ax.set_title("\n".join(wrap(
                        'Using ' + steps_encoding_method + ' & ' + class_generation_method + ', predicted with ' + regressor_name,
                        90)), fontsize=10)
                    fig.tight_layout()
                    dataset_name = os.path.basename(os.path.normpath(args.parent_folder))
                    plt.suptitle(dataset_name + ' (train)')
                    fig.subplots_adjust(top=0.88)
                    plt.savefig(os.path.join(output_path,
                                             'Train_' + metric + '_' + dataset_name + '_' + steps_encoding_method + '_' + class_generation_method + '_' + regressor_name + '.png'))
                    # ===========================
