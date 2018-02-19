import pandas as pd
import numpy as np
import glob


stroke_names = ["LB", "UT"] # Used for stroke data



def read_dataset(path_to_examples, sequence_length = 0, num_classes = 0, is_stroke= False):
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
        if not is_stroke: # If we are not looking at stroke data
            for csv_file in csv_file_path:
                if 'Axivity' in csv_file:
                    training_examples_csv.append(csv_file)
                else:
                    training_labels_csv.append(csv_file)
        else: # If we are looking at stroke data
            for csv_file in csv_file_path:
                if 'labels' in csv_file:
                    training_labels_csv.append(csv_file)
                else:
                    for name in stroke_names:
                        if name in csv_file:
                            training_examples_csv.append(csv_file)

    training_examples, training_lables = generate_examples_and_labels(training_examples_csv, training_labels_csv, is_stroke)
    if is_stroke:
        #print training_examples.shape
        delete_list = []
        for i in range(len(training_lables)):
            if training_lables[i] > 19:
                delete_list.append(i)
        training_lables = np.delete(training_lables, delete_list, axis=0)
        training_examples = np.delete(training_examples, delete_list, axis=0)
        print "Deleted: " + str(len(delete_list)) + " number of elements from " +path_to_examples
        #print training_examples.shape


    return training_examples, training_lables

def generate_examples_and_labels(examples_csv, labels_csv, is_stroke):
    training_examples_list = []

    training_examples = 0
    if not is_stroke:
        # Iterates over every second element in the training_examples_csv list
        for i in [x for x in range(len(examples_csv)-1) if x% 2 == 0]:
            if "THIGH" in examples_csv[i]:
                thigh = examples_csv[i]
                back = examples_csv[i + 1]
            else:
                thigh = examples_csv[i + 1]
                back = examples_csv[i]
                # Read both thigh and back datafiles
            training_examples_list.append(read_csv_file([thigh, back], examples=True))
        # Iterates over all the elements training_lables_csv list
        training_labels_list = [read_csv_file(training_label, False) for training_label in labels_csv]

        training_examples = []
        # Convert from dataframe to numpy array and append to list
        for i in range(len(training_examples_list)):
            training_examples.append(training_examples_list[i].values)

        # Go over training_examples list and convert it into a single numpy array
        training_examples = np.concatenate(training_examples)
    else:
        # Iterates over every second element in the training_examples_csv list
        for i in [x for x in range(len(examples_csv) - 1) if x % 2 == 0]:
            if "LB" in examples_csv[i]:
                back = examples_csv[i]
                thigh = examples_csv[i + 1]
            else:
                thigh = examples_csv[i]
                back = examples_csv[i+1]
            training_examples_list.append(read_csv_file([thigh, back], examples=True))

        training_examples = []
        # Convert from dataframe to numpy array and append to list
        for i in range(len(training_examples_list)):
            training_examples.append(training_examples_list[i].values)

        # Go over training_examples list and convert it into a single numpy array
        training_examples = np.concatenate(training_examples)

    training_labels = []

    # Iterates over all the elements training_lables_csv list
    training_labels_list = [read_csv_file(training_label, examples=False) for training_label in labels_csv]

    # Convert from dataframe to numpy array and append to list
    for i in range(len(training_labels_list)):
        training_labels.append(training_labels_list[i].values)

    # Go over training_labels list and convert into a single numpy array
    training_labels = np.concatenate(training_labels)


    training_labels = np.asarray(training_labels)
    return training_examples, training_labels

def read_csv_file(path_to_file, examples):
    if examples:
        training_THIGH = pd.read_csv(path_to_file[0])
        training_BACK = pd.read_csv(path_to_file[1])
        training_example = pd.concat([training_THIGH, training_BACK], axis=1)
    else:
        training_example = pd.read_csv(path_to_file)
    return training_example


