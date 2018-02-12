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

def decimate_data(data, indexes, downsampling_factor=2, filter_order=2):
    """
    Decimates the given dataframe using scipy's decimate method and returns a new decimated dataframe
    :param data: input dataframe
    :param indexes: indexes to use in the output dataframe
    :param downsampling_factor: the dowsnampling factor. Default is 2, since we want to halve the size of our data.
    :param filter_order: The order of the filter we want to use. Default is 2
    :return: dataframe of resampled data
    """
    # Get the name of the column we are using
    column_name = data.columns.values

    # Reshape dataframe to 1d array to prepare for decimation
    data = np.reshape(data.values, (np.product(data.values.shape),))

    # Decimate the signal using scipy
    data = sig.decimate(data, downsampling_factor, filter_order, "iir")

    # Convert decimated signal to dataframe again
    df = pd.DataFrame(data, index=indexes, columns=column_name)

    return df

def resample_data(data, indexes, downsampling_factor=2):
    """
    Resamples the given dataframe using scipy's resample method and returns a new resampled dataframe
    :param data: input dataFrame
    :param indexes: indexes to use for the output dataframe
    :param downsampling_factor: The factor of Haakon_downsampling. Deafault is 2, since we want to halve the size of our data
    :return: dataframe of resampled data
    """
    # Get the name of the column we are using
    column_name = data.columns.values

    # Reshape 1d dataframe to 1d array to prepare for resampling
    data = np.reshape(data.values, (np.product(data.values.shape),))

    # Calculate the number of samples needed
    number_of_samples = len(data)/downsampling_factor

    # Resample the signal using scipy
    data = sig.resample(data, number_of_samples)

    # Convert resampled signal to dataframe again
    df = pd.DataFrame(data, index=indexes, columns=column_name)

    return df

def resample_poly_data(data, indexes, downsampling_factor=2, up=1, down=2):
    """
        Resamples the given dataframe using scipy's resample_poly method and returns a new resampled dataframe
        :param data: input dataFrame
        :param indexes: indexes to use for the output dataframe
        :param downsampling_factor: The factor of Haakon_downsampling. Deafault is 2, since we want to halve the size of our data
        :param up: The up factor for scipy.signal.poly_resample
        :param down: The down factor for scipy.signal.poly_resample
        :return: dataframe of resampled data
        """
    # Get the name of the column we are using
    column_name = data.columns.values

    # Reshape dataframe to 1d array to prepare for resampling
    data = np.reshape(data.values, (np.product(data.values.shape),))

    # Resample the signal using scipy
    data = sig.resample_poly(data, up=up, down=down)

    # Convert resampled signal to dataframe again
    df = pd.DataFrame(data, index=indexes, columns=column_name)

    return df

def sample_every_nth(data, indexes, downsampling_factor=2):
    """
    Samples every other data point and returns a dataframe of these datapoints
    :param data: input dataFrame
    :param indexes: indexes to use for the output dataframe
    :param downsampling_factor: The factor of Haakon_downsampling. Deafault is 2, since we want to halve the size of our data
    :return: dataframe of resampled data
    """
    # Get the name of the column we are using
    column_name = data.columns.values

    # Reshape dataframe to 1d array
    data = np.reshape(data.values, (np.product(data.values.shape),))

    # Select every other element from the data and convert the final array to a list
    new_data = data[::downsampling_factor].tolist()

    # Build a dataframe from the list, using our set of indexes and the original column name
    df = pd.DataFrame(new_data, index=indexes,columns=column_name)

    return df

def average_two_and_two(data, indexes):
    # Get the name of the column we are using
    column_name = data.columns.values

    # Reshape dataframe to 1d array
    data = np.reshape(data.values, (np.product(data.values.shape),))
    new_data = np.zeros(data.shape[0]/2)

    iterator= 0
    for i in [x for x in range(len(data) - 1) if x % 2 == 0]:
        new_data[iterator] = (data[i]+data[i+1])/float(2)
        iterator +=1

    # Build a dataframe from the list, using our set of indexes and the original column name
    df = pd.DataFrame(new_data, index=indexes, columns=column_name)

    return df



def combine_dataframes(dataframes, names):
    """
    Combines a set of dataframes into one dataframe by concatenation
    :param dataframes: A list of dataframes to combine
    :param names: A list of names for the individual dataframes
    :return: A combinded dataframe with the given names
    """
    # Concatenate the list of dataframes
    combined_dataframe = pd.concat(dataframes, axis=1)

    # Set new names for the columns in the combined dataframe
    combined_dataframe.columns = names
    return combined_dataframe

"""
# Read a dataset
dataframe50 = rpdata.select_columns(rpdata.read_csv_test_data("/home/guest/Documents/HUNT4-data/50-100-Hz/exported-csv/7050_samplerange.csv"), [1])
dataframe100 = rpdata.select_columns(rpdata.read_csv_test_data("/home/guest/Documents/HUNT4-data/50-100-Hz/exported-csv/7100_samplerange.csv"), [1])
old, indexes = get_indexes(dataframe100)

dataframe50 = sample_every_nth(dataframe50, indexes, downsampling_factor=1)
#dataframe50 = rpdata.stretch_dataframe(dataframe50, old)

# Select only a single column from the loaded dataset


print "100Hz"
print dataframe100.mean()
print dataframe100.var()
print ""

print "50Hz"
print dataframe50.mean()
print dataframe50.var()
print ""


# Decimate data
decimated_data = decimate_data(dataframe100.copy(), indexes, 2, 1)
print "Decimated"
print decimated_data.mean()
print decimated_data.var()
print ""
# Stretch decimated data for plotting purposes
#decimated_data = rpdata.stretch_dataframe(decimated_data, old)

# Generate resampled data
resampled_data = resample_data(dataframe100.copy(), indexes)
print "Resampled"
print resampled_data.mean()
print resampled_data.var()
print ""
# Stretch resampled data for plotting purposes
#resampled_data = rpdata.stretch_dataframe(resampled_data, old)

# Generate resample_poly data
#resampled_poly_data = resample_poly_data(dataframe100.copy(), indexes, up=1, down=2)
#print "Resampled Poly"
#print resampled_poly_data.mean()
#print resampled_poly_data.var()
#print ""
# Stretch resampled data for plotting purposes
#resampled_poly_data = rpdata.stretch_dataframe(resampled_poly_data, old)


# Generate a resamnpled dataset using every other data point in the original set
naive_ds = sample_every_nth(dataframe100.copy(), indexes)
print "Every other"
print naive_ds.mean()
print naive_ds.var()
# Stretch resampled data for plotting purposes
#naive_ds = rpdata.stretch_dataframe(naive_ds, old)

# Combine the dataframes we want to visualize into one dataframe
#combined_dataframe = combine_dataframes([dataframe50, resampled_data], ["50Hz", "resampled"])


# Plot the final dataframe
#rpdata.plot_data(combined_dataframe)
"""