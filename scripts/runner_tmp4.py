"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""
import glob
import os

if __name__ == "__main__":
    """
    Allows to sequentially launch any number of scripts to generate results.
    """

    bins_to_explore = [2, 4, 8, 16, 32]
    split_method = 'equal_freq'
    output_classes = 'inside_bin'
    model = 'RandomForest'
    log_lvl = 'warning'
    n_jobs = -1

    datasets_directories = [f.path for f in os.scandir('../data/cleaned/') if f.is_dir()]
    datasets_names = [dataset_directory.split('/')[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    indexes_paths = [glob.glob(dataset_directory + '/*.index')[0] for dataset_directory in datasets_directories]
    indexes_paths = sorted(indexes_paths)  # Sort alphabetically

    cmd_list = []
    for dataset_name, index_path in zip(datasets_names, indexes_paths):
        goal_index = None
        with open(index_path) as f:
            goal_index = int(f.readline())

        for bins in bins_to_explore:
            cmd_list.append("python data_processing.py --dataset_path=\"../data/cleaned/{}/data.csv\" --goal_var_index=\"{}\" --n_bins=\"{}\" --output_classes=\"{}\" --split_method=\"{}\" --log_lvl=\"{}\"".format(dataset_name, goal_index, bins, output_classes, split_method, log_lvl))
            cmd_list.append("python feature_extraction.py --dataset_folder=\"../data/processed/{}/{}_bins_{}_{}/\" --classifier=\"{}\" --log_lvl=\"{}\" --n_jobs={}".format(dataset_name, bins, split_method, output_classes, model, log_lvl, n_jobs))
            cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/extracted_features/{}/{}_bins_{}_{}/{}_classifier\" --regressor=\"{}\" --log_lvl=\"{}\" --n_jobs={}".format(dataset_name, bins, split_method, output_classes, model, model, log_lvl, n_jobs))
            cmd_list.append("python compute_metrics.py  --predictions_folder=\"../data/predictions/{}/{}_bins_{}_{}/{}_classifier/{}_regressor\" --log_lvl=\"{}\"".format(dataset_name, bins, split_method, output_classes, model, model, 'info'))

        cmd_list.append("python generate_predictions.py --dataset_folder=\"../data/processed/{}/2_bins_{}_{}/\" --regressor=\"{}\" --log_lvl=\"{}\" --n_jobs={}".format(dataset_name, split_method, output_classes, model, log_lvl, n_jobs))
        cmd_list.append("python compute_metrics.py --predictions_folder=\"../data/predictions/{}/2_bins_{}_{}/Standard/{}\" --log_lvl=\"{}\"".format(dataset_name, split_method, output_classes, model + '_regressor', log_lvl))
        cmd_list.append("python visualisation.py --parent_folder=\"../data/metrics/{}\" --metric=\"RMSE\"".format(dataset_name))

    for c in cmd_list:
        print("\nLaunching : " + str(c))
        os.system(c)
