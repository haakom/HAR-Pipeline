import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import read_and_plot_data as rpdata
import fetch_dataframe_stats as dfstats
import matplotlib.mlab as mlab
import math
from downsample_signal import get_indexes, sample_every_nth, decimate_data, resample_data, average_two_and_two

hundred_means = []
hundred_vars = []

fifty_means = []
fifty_vars = []

fifty_mean_diff = []
fifty_var_diff = []

dec_mean_diff = []
dec_var_diff = []
dec_means = []
dec_vars = []

res_mean_diff = []
res_var_diff = []
res_means = []
res_vars = []

naive_mean_diff = []
naive_var_diff = []
naive_means = []
naive_vars = []

avg_mean_diff = []
avg_var_diff = []
avg_means = []
avg_vars = []

def norm_dist_differences(columns, data_range, plot=False, fifty = True, decimate = True, resample = True, average = True, every_second = True):
    means_return_list = []
    sigmas_return_list = []
    labels_return_list= []
    for i in range(columns[0], columns[1]):

        if data_range != None:
            dataframe100 = rpdata.select_columns(rpdata.read_csv_test_data(
                "/home/guest/Documents/HUNT4-data/50-100-Hz/exported-csv/7100_samplerange.csv"), [i+1])
            dataframe100 = dataframe100[0:data_range]

        else:
            dataframe100 = rpdata.select_columns(rpdata.read_csv_test_data(
                "/home/guest/Documents/HUNT4-data/50-100-Hz/exported-csv/7100_samplerange.csv"), [i + 1])
        old, indexes = get_indexes(dataframe100)


        hundred_mean, hundred_var = dfstats.get_mean_and_var(dataframe100[i+1])
        hundred_means.append(hundred_mean)
        hundred_vars.append(hundred_var)

        if fifty:
            if data_range != None:
                dataframe50 = rpdata.select_columns(rpdata.read_csv_test_data(
                    "/home/guest/Documents/HUNT4-data/50-100-Hz/exported-csv/7050_samplerange.csv"), [i + 1])
                dataframe50 = dataframe50[0:int(data_range/2)]

            else:
                dataframe50 = rpdata.select_columns(rpdata.read_csv_test_data(
                    "/home/guest/Documents/HUNT4-data/50-100-Hz/exported-csv/7050_samplerange.csv"), [i + 1])
            fifty_mean, fifty_var = dfstats.get_mean_and_var(dataframe50[i + 1])
            fifty_means.append(fifty_mean)
            fifty_vars.append(fifty_var)
            fifty_mean_diff.append(np.abs(hundred_mean - fifty_mean))
            fifty_var_diff.append(np.abs(hundred_var - fifty_var))

        if decimate:
            dec_data = decimate_data(dataframe100.copy(), indexes, 2, 2)
            dec_mean, dec_var = dfstats.get_mean_and_var(dec_data[i + 1])
            dec_means.append(dec_mean)
            dec_vars.append(dec_var)
            dec_mean_diff.append(np.abs(hundred_mean - dec_mean))
            dec_var_diff.append(np.abs(hundred_var - dec_var))

        if resample:
            res_data = resample_data(dataframe100.copy(), indexes)
            res_mean, res_var = dfstats.get_mean_and_var(res_data[i + 1])
            res_means.append(res_mean)
            res_vars.append(res_var)
            res_mean_diff.append(np.abs(hundred_mean- res_mean))
            res_var_diff.append(np.abs(hundred_var - res_var))

        if average:
            avg_data = average_two_and_two(dataframe100.copy(), indexes)
            avg_mean, avg_var = dfstats.get_mean_and_var(avg_data[i + 1])
            avg_means.append(avg_mean)
            avg_vars.append(avg_var)
            avg_mean_diff.append(np.abs(hundred_mean - avg_mean))
            avg_var_diff.append(np.abs(hundred_var - avg_var))

        if every_second:
            naive_data = sample_every_nth(dataframe100.copy(), indexes)
            naive_mean, naive_var = dfstats.get_mean_and_var(naive_data[i + 1])
            naive_means.append(naive_mean)
            naive_vars.append(naive_var)
            naive_mean_diff.append(np.abs(hundred_mean - naive_mean))
            naive_var_diff.append(np.abs(hundred_var - naive_var))


    if columns[1] == 6:
        x = np.arange(-1, 1, 0.01)
    else:
        x = np.arange(-0.5, 0.5, 0.01)

    if fifty:
        fifty_mean = sum(fifty_mean_diff) / float(len(fifty_mean_diff))
        fifty_var = sum(fifty_var_diff) / float(len(fifty_var_diff))
        fifty_sigma = math.sqrt(fifty_var)
        if plot:
            fifty = plt.plot(x, mlab.normpdf(x, fifty_mean, fifty_sigma), label="50Hz")
        means_return_list.append(fifty_mean)
        sigmas_return_list.append(fifty_sigma)
        labels_return_list.append("50Hz_")
        print "50Hz mean diff: " + str(fifty_mean)
        print "50Hz var diff: " + str(fifty_var)
        print " "

    if decimate:
        dec_mean = sum(dec_mean_diff) / float(len(dec_mean_diff))
        dec_var = sum(dec_var_diff) / float(len(dec_var_diff))
        dec_sigma = math.sqrt(dec_var)
        if plot:
            dec = plt.plot(x, mlab.normpdf(x, dec_mean, dec_sigma), label="Decimated")
        means_return_list.append(dec_mean)
        sigmas_return_list.append(dec_sigma)
        labels_return_list.append("Decimate_")
        print "Dec mean diff: " + str(dec_mean)
        print "Dec var diff: " + str(dec_var)
        print " "

    if resample:
        res_mean = sum(res_mean_diff) / float(len(res_mean_diff))
        res_var = sum(res_var_diff) / float(len(res_var_diff))
        res_sigma = math.sqrt(res_var)
        if plot:
            res = plt.plot(x, mlab.normpdf(x, res_mean, res_sigma), label="Resampled")
        means_return_list.append(res_mean)
        sigmas_return_list.append(res_sigma)
        labels_return_list.append("Resample_")
        print "Res mean diff: " + str(res_mean)
        print "Res var diff: " + str(res_var)
        print " "

    if average:
        avg_mean = sum(avg_mean_diff) / float(len(avg_mean_diff))
        avg_var = sum(avg_var_diff) / float(len(avg_var_diff))
        avg_sigma = math.sqrt(avg_var)
        if plot:
            avg = plt.plot(x, mlab.normpdf(x, avg_mean, avg_sigma), label="Averaged")
        means_return_list.append(avg_mean)
        sigmas_return_list.append(avg_sigma)
        labels_return_list.append("Average_")
        print "Avg mean diff: " + str(avg_mean)
        print "Avg var diff: " + str(avg_var)
        print " "

    if every_second:
        naive_mean = sum(naive_mean_diff) / float(len(naive_mean_diff))
        naive_var = sum(naive_var_diff) / float(len(naive_var_diff))
        naive_sigma = math.sqrt(naive_var)
        if plot:
            naive_data = plt.plot(x, mlab.normpdf(x, naive_mean, naive_sigma), label="Every_2nd")
        means_return_list.append(naive_mean)
        sigmas_return_list.append(naive_sigma)
        labels_return_list.append("Every_2nd_")
        print "Naive mean diff: " + str(naive_mean)
        print "Naive var diff: " + str(naive_var)

    if plot:
        plt.title("Normal distribution of difference from the 100Hz data")
        plt.legend()
        plt.show()

    return means_return_list, sigmas_return_list, labels_return_list

def norm_dists(columns, hundred = True, fifty = True, decimate = True, resample = True, average = True, every_second = True):
    for i in range(columns[0], columns[1]):

        dataframe100 = rpdata.select_columns(rpdata.read_csv_test_data(
            "/home/guest/Documents/HUNT4-data/50-100-Hz/exported-csv/7100_samplerange.csv"), [i + 1])
        old, indexes = get_indexes(dataframe100)

        if hundred:
            hundred_mean, hundred_var = dfstats.get_mean_and_var(dataframe100[i + 1])
            hundred_means.append(hundred_mean)
            hundred_vars.append(hundred_var)

        if fifty:
            dataframe50 = rpdata.select_columns(rpdata.read_csv_test_data(
                "/home/guest/Documents/HUNT4-data/50-100-Hz/exported-csv/7050_samplerange.csv"), [i + 1])
            fifty_mean, fifty_var = dfstats.get_mean_and_var(dataframe50[i + 1])
            fifty_means.append(fifty_mean)
            fifty_vars.append(fifty_var)


        if decimate:
            dec_data = decimate_data(dataframe100.copy(), indexes, 2, 2)
            dec_mean, dec_var = dfstats.get_mean_and_var(dec_data[i + 1])
            dec_means.append(dec_mean)
            dec_vars.append(dec_var)


        if resample:
            res_data = resample_data(dataframe100.copy(), indexes)
            res_mean, res_var = dfstats.get_mean_and_var(res_data[i + 1])
            res_means.append(res_mean)
            res_vars.append(res_var)


        if average:
            avg_data = average_two_and_two(dataframe100.copy(), indexes)
            avg_mean, avg_var = dfstats.get_mean_and_var(avg_data[i + 1])
            avg_means.append(avg_mean)
            avg_vars.append(avg_var)


        if every_second:
            naive_data = sample_every_nth(dataframe100.copy(), indexes)
            naive_mean, naive_var = dfstats.get_mean_and_var(naive_data[i + 1])
            naive_means.append(naive_mean)
            naive_vars.append(naive_var)

    x = np.arange(-2.5, 2, 0.01)

    if hundred:
        hundred_actual_mean = sum(hundred_means) / float(len(hundred_means))
        hundred_actual_var = sum(hundred_vars) / float(len(hundred_vars))
        hundred_sigma = math.sqrt(hundred_actual_var)
        hundred = plt.plot(x, mlab.normpdf(x, hundred_actual_mean, hundred_sigma), label="100Hz")

    if fifty:
        fifty_actual_mean = sum(fifty_means) / float(len(fifty_means))
        fifty_actual_var = sum(fifty_vars) / float(len(fifty_vars))
        fifty_sigma = math.sqrt(fifty_actual_var)
        fifty = plt.plot(x, mlab.normpdf(x, fifty_actual_mean, fifty_sigma), label="50Hz")

    if decimate:
        dec_actual_mean = sum(dec_means) / float(len(dec_means))
        dec_actual_var = sum(dec_vars) / float(len(dec_vars))
        dec_sigma = math.sqrt(dec_actual_var)
        dec = plt.plot(x, mlab.normpdf(x, dec_actual_mean, dec_sigma), label="Decimated")

    if resample:
        res_actual_mean = sum(res_means) / float(len(res_means))
        res_actual_var = sum(res_vars) / float(len(res_vars))
        res_sigma = math.sqrt(res_actual_var)
        res = plt.plot(x, mlab.normpdf(x, res_actual_mean, res_sigma), label="Resampled")

    if average:
        avg_actual_mean = sum(avg_means) / float(len(avg_means))
        avg_actual_var = sum(avg_vars) / float(len(avg_vars))
        avg_sigma = math.sqrt(avg_actual_var)
        avg = plt.plot(x, mlab.normpdf(x, avg_actual_mean, avg_sigma), label="Averaged")

    if every_second:
        naive_actual_mean = sum(naive_means) / float(len(naive_means))
        naive_actual_var = sum(naive_vars) / float(len(naive_vars))
        naive_sigma = math.sqrt(naive_actual_var)
        naive_data = plt.plot(x, mlab.normpdf(x, naive_actual_mean, naive_sigma), label="Every_other")


    plt.title("Normal distribution of 100Hz and downsampled data")
    plt.legend()
    plt.show()



