"""
Orange Labs
Authors : Colin Troisemaine & Vincent Lemaire
contact : colin.troisemaine@gmail.com
"""

import argparse
import pandas as pd
import time


def argument_parser():
    """
    A parser to allow user to easily process any dataset using the split method and class generation he desires.
    """

    parser = argparse.ArgumentParser(usage='\n python3 data_processing.py [dataset_path] [output_path] [split_method] '
                                           '[output_classes] [delimiter] [header] [decimal] [na_values]'
                                           '\n Example : python scripts/data_processing.py '
                                           '--dataset_path=\"../data/raw/3D_Road_Network_Dataset/3D_spatial_network'
                                           '.csv\"',
                                     description="This program allows to process datasets to be used later.")

    parser.add_argument('--dataset_path',
                        type=str,
                        help='The dataset to process',
                        required=True)

    parser.add_argument('--output_path',
                        type=str,
                        default='../data/processed/',
                        help='The folder where the result will be written')

    parser.add_argument('--split_method',
                        type=str,
                        default="equal_width",
                        choices=["equal_width", "equal_fred", "kmeans"],
                        help='The splitting method to use')

    parser.add_argument('--output_classes',
                        type=str,
                        default="below_threshold",
                        choices=["below_threshold", "inside_bin"],
                        help='The method of class generation')

    parser.add_argument('--delimiter',
                        type=str,
                        default=",",
                        help='Delimiter to use when reading the dataset')

    parser.add_argument('--header',
                        type=str,
                        default="infer",
                        choices=["infer", "None"],
                        help='Infer the column names or use None if the first line isn\' a csv header line')

    parser.add_argument('--decimal',
                        type=str,
                        default=",",
                        help='Character to recognize as decimal point')

    parser.add_argument('--na_values',
                        type=str,
                        help='Additional string to recognize as NA/NaN')

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    print("Reading the dataset's file...")
    reading_start_time = time.time()
    imported_dataset = pd.read_csv(args.dataset_path, delimiter=args.delimiter, header=args.header,
                                   decimal=args.decimal, na_values=args.na_values)
    print("Dataset imported ({0:.2f}".format(time.time() - reading_start_time) + "sec)")

    print("\nDataset's first 3 rows :")
    print(imported_dataset.head(3))

    # TODO

    pass
