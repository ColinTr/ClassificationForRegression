"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""

import argparse
import time
import os


def argument_parser():
    """
    A parser to allow user to easily generate the results of any number of classifiers using our method.
    """

    parser = argparse.ArgumentParser(usage='\n python runner.py [regressor] [classifier]',
                                     description=".")

    parser.add_argument('--classifier',
                        type=str,
                        help='The classifier models to use',
                        choices=["RandomForest", "LogisticRegression", "XGBoost", "GaussianNB", "Khiops", "DecisionTree"],
                        required=True)

    parser.add_argument('--regressor',
                        type=str,
                        help='The regressor models to use',
                        choices=["RandomForest", "LinearRegression", "XGBoost", "GaussianNB", "Khiops", "DecisionTree"],
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """
    args = argument_parser()

    split_method = 'equal_freq'
    output_classes = 'below_threshold'
    log_lvl = 'info'
    n_jobs = -1
    n_bins = 32
    grid_search = 'True'

    datasets_directories = [f.path for f in os.scandir(os.path.join('..', 'data', 'cleaned')) if f.is_dir()]
    datasets_names = [dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    cmd_list = []
    for dataset_name in datasets_names:
        print('============== ' + str(dataset_name) + ' :')
        #os.system("python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --n_bins=\"{}\" --output_classes=\"{}\" --split_method=\"{}\" --log_lvl=\"{}\""
        #          .format(dataset_name, n_bins, output_classes, split_method, log_lvl))
        os.system("python feature_extraction.py --dataset_folder=\"../data/processed/{}/{}_bins_{}_{}/\" --classifier=\"{}\" --log_lvl=\"{}\" --n_jobs={}"
                  .format(dataset_name, n_bins, split_method, output_classes, args.classifier, 'warning', n_jobs))

        print('= Normal version :')
        start_time = time.time()
        os.system("python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/{}_bins_{}_{}/{}_classifier\" --regressor {} --log_lvl {} --n_jobs={} --grid_search {}"
                  .format(dataset_name, n_bins, split_method, output_classes, args.classifier, args.regressor, 'warning', n_jobs, grid_search))
        os.system("python compute_metrics.py --predictions_folder=\"../data/predictions/{}/{}_bins_{}_{}/{}/{}/\" --log_lvl=\"{}\""
                  .format(dataset_name, n_bins, split_method, output_classes, args.classifier + '_classifier', args.regressor + '_regressor', log_lvl))
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))

        print('= Only extended version :')
        start_time = time.time()
        os.system("python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/{}_bins_{}_{}/{}_classifier\" --regressor {} --log_lvl {} --n_jobs={} --grid_search {} --extracted_only True"
                  .format(dataset_name, n_bins, split_method, output_classes, args.classifier, args.regressor, 'warning', n_jobs, grid_search))
        os.system("python compute_metrics.py --predictions_folder=\"../data/predictions/{}/{}_bins_{}_{}/{}/{}/\" --log_lvl=\"{}\""
                  .format(dataset_name, n_bins, split_method, output_classes, args.classifier + '_classifier', args.regressor + '_regressor', log_lvl))
        print("Elapsed time : {0:.2f}".format(time.time() - start_time))
