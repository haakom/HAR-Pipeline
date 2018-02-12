import downsampling_data_distributions as ddd
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import Haakon_Recurrent_ANN.read_data as rd
import math
import numpy as np

def plot_downsample_norm_dists(data_range= None, fifty=False, decimate=False, resample=False, average=False, every_second=False):
    one_to_three_means, one_to_three_sigmas, labels = ddd.norm_dist_differences([0, 3], data_range=data_range, fifty=fifty, decimate=decimate, resample=resample, average=average, every_second=every_second)
    four_to_six_means, four_to_six_sigmas, labels  = ddd.norm_dist_differences([3, 6], data_range=data_range, fifty=fifty, decimate=decimate, resample=resample, average=average, every_second=every_second)

    x = np.arange(-1, 1, 0.01)

    for i in range(len(one_to_three_sigmas)):
        plt.plot(x, mlab.normpdf(x, one_to_three_means[i], one_to_three_sigmas[i]), label=labels[i]+"BACK?")
        plt.plot(x, mlab.normpdf(x, four_to_six_means[i], four_to_six_sigmas[i]), label=labels[i]+"THIGH?")

    plt.axvline(x=0, linestyle='--', color="Red")
    plt.legend()
    plt.show()

def plot_data_norm_dists(paths_to_data_folders):
    data_points_list = read_datapoints_to_list(paths_to_data_folders)
    print data_points_list[0].shape

    normal_mean_THIGH, normal_mean_BACK = get_sensor_mean(data_points_list[0])
    normal_var_THIGH, normal_var_BACK = get_sensor_variance(data_points_list[0])
    normal_sigma_THIGH = math.sqrt(normal_var_THIGH)
    normal_sigma_BACK = math.sqrt(normal_var_BACK)


    in_lab_mean_THIGH, in_lab_mean_BACK = get_sensor_mean(data_points_list[1])
    in_lab_var_THIGH, in_lab_var_BACK  = get_sensor_variance(data_points_list[1])
    in_lab_sigma_THIGH = math.sqrt(in_lab_var_THIGH)
    in_lab_sigma_BACK = math.sqrt(in_lab_var_BACK)

    x = np.arange(-2, 2, 0.01)

    plt.title("In Lab data vs Out of Lab data")
    plt.plot(x, mlab.normpdf(x, normal_mean_THIGH, normal_sigma_THIGH), label="out-of-lab_THIGH")
    plt.plot(x, mlab.normpdf(x, normal_mean_BACK, normal_sigma_BACK), label="out-of-lab_BACK")

    plt.plot(x, mlab.normpdf(x, in_lab_mean_THIGH, in_lab_sigma_THIGH), label="in-Lab_THIGH")
    plt.plot(x, mlab.normpdf(x, in_lab_mean_BACK, in_lab_sigma_BACK), label="in-Lab_BACK")
    plt.legend()
    plt.show()


def get_sensor_mean(datapoints):
    means = []
    for i in range(datapoints.shape[1]):
        means.append(np.mean(datapoints[:, i, :]))
    return means


def get_sensor_variance(datapoints):
    variances = []
    for i in range(datapoints.shape[1]):
        variances.append(np.var(datapoints[:, i, :]))

    return variances

def read_datapoints_to_list(list_of_folders):
    folder_datapoints = []

    for folder in list_of_folders:
        print folder
        datapoints, labels = rd.select_csv_files(folder)
        datapoints = rd.generate_examples_and_labels(datapoints, labels, 1, 20, print_stats=True, generate_labels=False)
        datapoints = np.reshape(datapoints, newshape=[datapoints.shape[0], datapoints.shape[2]])
        datapoints = np.reshape(datapoints,
                                newshape=[datapoints.shape[0], datapoints.shape[1] / (datapoints.shape[1] / 2),
                                          datapoints.shape[1] / 2])
        folder_datapoints.append(datapoints)

    return folder_datapoints

plot_data_norm_dists(["/home/guest/Documents/HAR-Pipeline/DATA/trene", "/home/guest/Documents/HAR-Pipeline/DATA/HUNT4-Training-Data-InLab-UpperBackThigh"])

