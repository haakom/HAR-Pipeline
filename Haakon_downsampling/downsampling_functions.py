from scipy import signal as sig
import numpy as np
import pandas as pd
import read_and_plot_data as rpdata

def get_indexes(data, downsampling_factor=2):
    """
    Creates a list of indexes, using the original indexes and the Haakon_downsampling factor. It selects every n'th element
    from the original index list to create appropriate indexes for the downsampled dataframes created in the other
    methods in this document.
    :param data: input dataframe
    :param downsampling_factor: the Haakon_downsampling factor we are using. Default is 2
    :return: List, List: A list of the original indexes, A list of the new indexes
    """
    # Find the original indexes
    indexes = data.index.values

    # Extract every n'th index using the Haakon_downsampling factor
    new_indexes = indexes[::downsampling_factor].tolist()

    return indexes, new_indexes

def decimate_data(data, indexes=None, downsampling_factor=2, filter_order=2):
    """
    Decimates the given dataframe using scipy's decimate method and returns a new decimated dataframe
    :param data: input dataframe
    :param indexes: indexes to use in the output dataframe
    :param downsampling_factor: the dowsnampling factor. Default is 2, since we want to halve the size of our data.
    :param filter_order: The order of the filter we want to use. Default is 2
    :return: dataframe of resampled data
    """

    # Reshape dataframe to 1d array to prepare for decimation
    data = data.values

    # Decimate the signal using scipy
    data = sig.decimate(data, downsampling_factor, filter_order, "iir")

    if indexes == None:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(data)
    else:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(data, index=indexes)

    return df

def resample_data(data, indexes=None, downsampling_factor=2):
    """
    Resamples the given dataframe using scipy's resample method and returns a new resampled dataframe
    :param data: input dataFrame
    :param indexes: indexes to use for the output dataframe
    :param downsampling_factor: The factor of Haakon_downsampling. Deafault is 2, since we want to halve the size of our data
    :return: dataframe of resampled data
    """

    # Reshape 1d dataframe to 1d array to prepare for resampling
    data = data.values

    # Calculate the number of samples needed
    number_of_samples = len(data)/downsampling_factor

    # Resample the signal using scipy
    data = sig.resample(data, number_of_samples)

    if indexes == None:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(data)
    else:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(data, index=indexes)

    return df

def resample_poly_data(data, indexes=None, downsampling_factor=2, up=1, down=2):
    """
        Resamples the given dataframe using scipy's resample_poly method and returns a new resampled dataframe
        :param data: input dataFrame
        :param indexes: indexes to use for the output dataframe
        :param downsampling_factor: The factor of Haakon_downsampling. Deafault is 2, since we want to halve the size of our data
        :param up: The up factor for scipy.signal.poly_resample
        :param down: The down factor for scipy.signal.poly_resample
        :return: dataframe of resampled data
        """


    # Reshape dataframe to 1d array to prepare for resampling
    data = data.values

    # Resample the signal using scipy
    data = sig.resample_poly(data, up=up, down=down)

    if indexes == None:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(data)
    else:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(data, index=indexes)

    return df

def sample_every_nth(data, indexes=None, downsampling_factor=2):
    """
    Samples every other data point and returns a dataframe of these datapoints
    :param data: input dataFrame
    :param indexes: indexes to use for the output dataframe
    :param downsampling_factor: The factor of Haakon_downsampling. Deafault is 2, since we want to halve the size of our data
    :return: dataframe of resampled data
    """

    # Reshape dataframe to 1d array
    data = data.values

    # Select every other element from the data and convert the final array to a list
    new_data = data[::downsampling_factor].tolist()

    if indexes == None:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(new_data)
    else:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(new_data, index=indexes)

    return df

def average_two_and_two(data, indexes=None):
    """
    Averages the value of two and two indexes to halve the number of of data points
    :param data: dataframe (single column)
    :param indexes: indexes: indexes to use for the output dataframe
    :return: dataframe of averaged data
    """

    # Reshape dataframe to 1d array
    data = data.values
    new_data = np.zeros(data.shape[0]/2, dtype=np.float64)

    iterator= 0
    for i in [x for x in range(len(data)) if x % 2 == 0]:
        #print i
        new_data[iterator] = (np.float64(data[i])+np.float64(data[i+1]))/np.float64(2)
        #print str(data[i]) + str(data[i+1]) + ": " +str(new_data[iterator])


        iterator +=1

    if indexes == None:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(new_data)
    else:
        # Build a dataframe from the list, using our set of indexes and the original column name
        df = pd.DataFrame(new_data, index=indexes)

    return df

def combine_dataframes(dataframes, names=None):
    """
    Combines a set of dataframes into one dataframe by concatenation
    :param dataframes: A list of dataframes to combine
    :param names: A list of names for the individual dataframes
    :return: A combinded dataframe with the given names
    """
    # Concatenate the list of dataframes
    combined_dataframe = pd.concat(dataframes, axis=1)

    # Set new names for the columns in the combined dataframe
    if names != None:
        combined_dataframe.columns = names
    return combined_dataframe

def do_downsampling(dataframe, downsampling_function):
    single_column_dataframes = []
    for i in range(len(dataframe.columns)):
        single_column_dataframes.append(
            downsampling_function(
                dataframe[dataframe.columns[i]]))

    return combine_dataframes(single_column_dataframes)
