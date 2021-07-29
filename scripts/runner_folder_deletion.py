import os
import shutil

folder_path = os.path.join('D:\\', 'metrics_vm_ubuntu')

if os.path.exists(folder_path):

    datasets_directories = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    datasets_names = [dataset_directory.split(os.path.sep)[-1] for dataset_directory in datasets_directories]
    datasets_names = sorted(datasets_names)

    for dataset_name in datasets_names:
        bins_level_directories = [f.path for f in os.scandir(os.path.join(folder_path, dataset_name)) if f.is_dir()]
        bins_level_directories = [bins_level_directory.split(os.path.sep)[-1] for bins_level_directory in bins_level_directories]

        for bins_level_directory in bins_level_directories:
            classifier_level_directories = [f.path for f in os.scandir(os.path.join(folder_path, dataset_name, bins_level_directory)) if f.is_dir()]
            classifier_level_directories = [classifier_level_directory.split(os.path.sep)[-1] for classifier_level_directory in classifier_level_directories]

            for classifier_level_directory in classifier_level_directories:
                if classifier_level_directory != 'Khiops_classifier' and classifier_level_directory != 'Standard' :
                    try:
                        print('Deleting ' + str(os.path.join(folder_path, dataset_name, bins_level_directory, classifier_level_directory)) + '...')
                        shutil.rmtree(os.path.join(folder_path, dataset_name, bins_level_directory, classifier_level_directory))
                    except OSError as e:
                        print("Error: %s - %s." % (e.filename, e.strerror))
                else :
                    regressor_level_directories = [f.path for f in os.scandir(os.path.join(folder_path, dataset_name, bins_level_directory, classifier_level_directory)) if f.is_dir()]
                    regressor_level_directories = [regressor_level_directory.split(os.path.sep)[-1] for regressor_level_directory in regressor_level_directories]

                    for regressor_level_directory in regressor_level_directories:
                        if regressor_level_directory != 'Khiops_regressor' :
                            try:
                                print('Deleting ' + str(os.path.join(folder_path, dataset_name, bins_level_directory, classifier_level_directory, regressor_level_directory)) + '...')
                                shutil.rmtree(os.path.join(folder_path, dataset_name, bins_level_directory, classifier_level_directory, regressor_level_directory))
                            except OSError as e:
                                print("Error: %s - %s." % (e.filename, e.strerror))

else:
    # file not found message
    print("File not found in the directory")