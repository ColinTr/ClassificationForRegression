"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import argparse
import os


def argument_parser():
    """
    A parser to allow user to easily generate the results of any number of classifiers using our method.
    """

    parser = argparse.ArgumentParser(usage='\n python runner.py [dataset_name] [goal_index] [classifiers]+',
                                     description="This program allows to run all the scripts necessary to generate the"
                                                 " final figures on any number of classifiers for a given dataset.")

    parser.add_argument('--dataset_name',
                        type=str,
                        help='The dataset to use',
                        required=True)

    parser.add_argument('--goal_index',
                        type=int,
                        help='The index of the goal variable',
                        required=True)

    parser.add_argument('--classifiers',
                        type=str,
                        nargs='+',  # 1 or more values expected
                        help='The classifiers to compare',
                        required=True,
                        choices=["RandomForest", "LogisticRegression", "XGBoost", "GaussianNB", "Khiops"])

    return parser.parse_args()


if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """

    args = argument_parser()

    bins_to_explore = [5, 10, 15, 20, 30, 40, 50]

    cmd_list = []

    # Process the dataset
    for bins in bins_to_explore:
        cmd_list.append("python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index={} --n_bins={} --log_lvl=warning".format(args.dataset_name, args.goal_index, bins))

    # Extract the features
    for classifier in args.classifiers:
        for bins in bins_to_explore:
            cmd_list.append("python feature_extraction.py --dataset_folder=\"../data/processed/{}/{}_bins_equal_freq_below_threshold/\" --classifier=\"{}\" --log_lvl=warning".format(args.dataset_name, bins, classifier))

    # Generate the predictions
    for classifier_1 in args.classifiers:
        for classifier_2 in args.classifiers:
            for bins in bins_to_explore:
                cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/{}_bins_equal_freq_below_threshold/{}/\" --regressor=\"{}\" --log_lvl=warning".format(args.dataset_name, bins, classifier_1 + '_classifier', classifier_2))

    # Compute the metrics
    for classifier_1 in args.classifiers:
        for classifier_2 in args.classifiers:
            for bins in bins_to_explore:
                cmd_list.append("python compute_metrics.py --predictions_folder=\"../data/predictions/{}/{}_bins_equal_freq_below_threshold/{}/{}/\" --log_lvl=warning".format(args.dataset_name, bins, classifier_1 + '_classifier', classifier_2 + '_regressor'))

    # Compute the baseline
    for classifier in args.classifiers:
        cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/processed/{}/5_bins_equal_freq_below_threshold/\" --regressor=\"{}\" --log_lvl=warning".format(args.dataset_name, classifier))
        cmd_list.append("python compute_metrics.py --predictions_folder=\"../data/predictions/{}/5_bins_equal_freq_below_threshold/Standard/{}\" --log_lvl=warning".format(args.dataset_name, classifier + '_regressor'))

    # Create the graphs
    cmd_list.append("python visualisation.py --parent_folder=\"../data/metrics/{}\"".format(args.dataset_name))

    for c in cmd_list:
        print("Launching :\n" + str(c))
        os.system(c)
