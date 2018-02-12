import pandas as pd
import numpy as np
import os
import glob
import math
from collections import Counter


def read_dataset(path_to_examples, sequence_length, num_classes):
    # Read all paths in specified folder
    example_folders_paths = glob.glob(path_to_examples + "/*")
    csv_file_paths = []

    # Select only files with .csv ending
    for example_folder in example_folders_paths:
        csv_file_paths.append(glob.glob(example_folder+"/*.csv"))

    # Split into example and label files
    training_examples_csv = []
    training_labels_csv = []
    for csv_file_path in csv_file_paths:
        for csv_file in csv_file_path:
            if 'Axivity' in csv_file:
                training_examples_csv.append(csv_file)
            else:
                training_labels_csv.append(csv_file)

    training_examples, training_lables = generate_examples_and_labels(training_examples_csv, training_labels_csv)


    return training_examples, training_lables

def generate_examples_and_labels(examples_csv, labels_csv):
    print examples_csv
    training_examples_list = []
    # Iterates over every second element in the training_examples_csv list
    for i in [x for x in range(len(examples_csv)-1) if x% 2 == 0]:
        # Read both thigh and back datafiles
        training_examples_list.append(read_csv_file([examples_csv[i], examples_csv[i + 1]], training = True))

    # Iterates over all the elements training_lables_csv list
    training_labels_list = [read_csv_file(training_label, False) for training_label in labels_csv]

    training_examples = []
    # Convert from dataframe to numpy array and append to list
    for i in range(len(training_examples_list)):
        training_examples.append(training_examples_list[i].values)

    # Go over training_examples list and convert it into a single numpy array
    training_examples = np.concatenate(training_examples)

    training_labels = []
    # Convert from dataframe to numpy array and append to list
    for i in range(len(training_labels_list)):
        training_labels.append(training_labels_list[i].values)

    # Go over training_labels list and convert into a single numpy array
    training_labels = np.concatenate(training_labels)


    training_labels = np.asarray(training_labels)
    return training_examples, training_labels




def read_csv_file(path_to_file, training):
    if training:
        #print path_to_file
        training_THIGH = pd.read_csv(path_to_file[0])
        training_BACK = pd.read_csv(path_to_file[1])
        training_example = pd.concat([training_THIGH, training_BACK], axis=1)
    else:
        #print path_to_file
        training_example = pd.read_csv(path_to_file)
        print path_to_file
        for col in training_example:
            print training_example[col].unique()
    return training_example


