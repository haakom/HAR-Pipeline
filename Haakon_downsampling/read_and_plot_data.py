from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def read_csv_data(filepath):
    """
    Takes in a csv file, reads it using pandas and returns the pandas dataframe
    :param filepath: path to the file we want to read
    :return: pandas dataframe
    """
    return pd.read_csv(filepath)

def read_csv_test_data(filepath):
    """
     Takes in a csv file, reads it using pandas and returns the pandas dataframe
     :param filepath: path to the file we want to read
     :return: pandas dataframe
     """
    return pd.DataFrame(pd.read_csv(filepath).values)

def select_data_slice(dataframe, slice_interval):
    """
    Slices the inputed dataframe and returns the data slice
    :param dataframe: dataFrame to be sliced
    :param slice_interval: The interval we want to use
    :return: The sliced dataFrame
    """
    return dataframe[slice_interval[0]:slice_interval[1]]

def select_columns(dataframe, columns):
    """
    Extracts the defined columns from the dataframe and returns them
    :param dataframe: input dataframe
    :param columns: a list of columns to use
    :return:
    """
    strcolumns = [dataframe.columns.values[column] for column in columns]
    return dataframe[strcolumns]

def stretch_dataframe(downsampled_dataframe, new_indexes):
   """
   Stretches the dataframe to fit the original dataframe size
   :param downsampled_dataframe: input dataframe (a downsampled dataframe)
   :param new_indexes: the indexes we want to use
   :return: a reindexed dataframe using the new indexes
   """
   return downsampled_dataframe.reindex(new_indexes)

def plot_data(dataframe, columns="All"):
    """
    Plots the data using matplotlib
    :param dataframe: the dataFrame to plot
    :param columns: List. The columns to be plotted, default is all columns
    :return: NOTHING
    """
    if "All" not in str(columns):
        columns_list = []
        dfcolumns= dataframe.columns.values
        for i in range(len(columns)):
            columns_list.append(dfcolumns[columns[i]])
        print columns_list
        dataframe[columns_list].plot(marker='x')
    else:
        dataframe.plot(marker='x')
    plt.show()
