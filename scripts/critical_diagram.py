"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
Maintainer : colin.troisemaine@gmail.com
"""
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import Orange


if __name__ == "__main__":
    n_bins = 32

    # The paths to the RMSE table where the results are stored. At least two tables must be given to be compared
    rmse_tables = [
        ('..\\data\\results_tables\\equal_freq_below_threshold_RandomForest_classifier_DecisionTree_regressor\\RMSE_table.csv', str(n_bins) + ' T Test RMSE'),
        ('..\\data\\results_tables\\equal_freq_below_threshold_RandomForest_classifier_DecisionTree_regressor\\RMSE_table.csv', 'Base Test RMSE'),
        ('..\\data\\results_tables\\equal_freq_below_threshold_RandomForest_classifier_LinearRegression_regressor\\RMSE_table.csv', str(n_bins) + ' T Test RMSE'),
        ('..\\data\\results_tables\\equal_freq_below_threshold_RandomForest_classifier_LinearRegression_regressor\\RMSE_table.csv', 'Base Test RMSE'),
        ('..\\data\\results_tables\\equal_freq_below_threshold_RandomForest_classifier_RandomForest_regressor\\RMSE_table.csv', str(n_bins) + ' T Test RMSE'),
        ('..\\data\\results_tables\\equal_freq_below_threshold_RandomForest_classifier_RandomForest_regressor\\RMSE_table.csv', 'Base Test RMSE')
    ]

    # The labels corresponding to the paths (in the same order)
    labels = ['DT+', 'DT', 'LR+', 'LR', 'RF+', 'RF']

    # The method that is compared with other methods If omitted, show pairwise comparison of methods
    cdmethod = None

    if len(labels) != len(rmse_tables):
        raise Exception("Please define as many labels as there are rmse_tables")

    groups = []
    # Get for each table the column corresponding to the parameters
    for path, column in rmse_tables:
        tmp_df = pd.read_csv(path)
        groups.append(list(tmp_df[column]))

    # Rank the data for each dataset
    ranked_data = [list(stats.rankdata(row)) for row in np.array(groups).T]

    average_ranks = [np.mean(vect) for vect in np.array(ranked_data).T]

    cd = Orange.evaluation.compute_CD(average_ranks, len(groups[0]))
    Orange.evaluation.graph_ranks(average_ranks, labels, cd=cd, width=6, textspace=1.5, cdmethod=cdmethod)
    plt.savefig('critical_diagram_' + str(n_bins) + 'bins.png')
