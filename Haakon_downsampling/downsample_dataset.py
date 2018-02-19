import pandas as pd
import numpy as np
import downsampling_functions as df
from Haakon_Recurrent_ANN import read_data as rd
import os

path_to_data = "/home/guest/Documents/dataset/"
store_data_path_builder = "/home/guest/Documents/Downsampled-data/"
downsampling_functions = [df.average_two_and_two,
                          df.sample_every_nth,df.resample_data]
downsampling_function_names = ["TWO_AND_TWO_AVG",
                              "EVERY_OTHER", "RESAMPLE"]


def do_examples(examples):
    for example in examples:
        new_name = example.strip(path_to_data)
        subject_name = new_name[0:3]
        if not os.path.isdir(store_data_path+"/"+subject_name):
            os.makedirs(store_data_path+"/"+subject_name)
        new_name =new_name[4:]
        #print subject_name
        #print new_name
        #print store_data_path+"/"+subject_name+"/"+new_name
        if ".csv.csv" in new_name:
            #print new_name
            new_name = new_name.replace(".csv.csv", ".csv")
        example_dataframe = df.do_downsampling(pd.read_csv(example), downsampling_function)
        example_dataframe.to_csv(store_data_path+"/"+subject_name+"/"+new_name, index=False, header=False)

def do_labels(labels):
    for label in labels:
        new_name = label.strip(path_to_data)
        subject_name = new_name[0:3]
        if not os.path.isdir(store_data_path+"/"+subject_name):
            os.makedirs(store_data_path+"/"+subject_name)
        new_name =new_name[4:]
        #print subject_name
        #print new_name
        #print store_data_path+"/"+subject_name+"/"+new_name
        if ".csv.csv" in new_name:
            #print new_name
            new_name = new_name.replace(".csv.csv", ".csv")
        label_dataframe = df.do_downsampling(pd.read_csv(label), df.sample_every_nth)
        label_dataframe.to_csv(store_data_path+"/"+subject_name+"/"+new_name, index=False, header=False)


examples, labels = rd.select_csv_files(path_to_data, in_lab=True)
print examples
print labels
iterator = 0
for downsampling_function in downsampling_functions:
    store_data_path = store_data_path_builder+downsampling_function_names[iterator]+ "/"
    if not os.path.isdir(store_data_path):
        os.makedirs(store_data_path)
    print downsampling_function_names[iterator]
    do_examples(examples)
    do_labels(labels)
    iterator +=1


