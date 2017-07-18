from __future__ import print_function, division

import numpy as np
from collections import Counter

import scipy.stats


def generate_all_integer_combinations(stop_integer):
    combinations_of_certain_length = dict()
    combinations_of_certain_length[1] = {(j,) for j in range(stop_integer)}

    for i in range(2, stop_integer + 1):
        combinations_of_certain_length[i] = set()
        for t in combinations_of_certain_length[i - 1]:
            largest_element = t[-1]
            for j in range(largest_element + 1, stop_integer):
                l = list(t)
                l.append(j)
                new_combination = tuple(l)
                combinations_of_certain_length[i].add(new_combination)

    values = [sorted(list(combinations_of_certain_length[j])) for j in range(2, stop_integer + 1)]

    reduced_list = []
    for value in values:
        reduced_list += value

    return reduced_list


def peak_acceleration(array):
    return max(np.linalg.norm(array, axis=1))


def max_and_mins(array):  # Something should be done about this, so it's the largest deviation or something
    return np.hstack(array.max())


def means_and_std_factory(absolute_values=False):
    def means_and_std(a):
        if absolute_values:
            a = abs(a)
        return np.hstack((np.mean(a, axis=0), np.std(a, axis=0)))

    return means_and_std


def most_frequent_value(array):
    if len(array.shape) > 1:
        most_common = []
        for column in array.T:
            counts = Counter(column)
            top = counts.most_common(1)[0][0]
            most_common.append(top)

        return np.array(most_common)

    counts = Counter(array)
    top = counts.most_common(1)[0][0]
    return np.array([top])


def column_product_factory(column_indices):
    def columns_product(array):
        transposed = np.transpose(array)[[column_indices]]  # Transpose for simpler logic

        product = np.ones(transposed.shape[1])
        for row in transposed:
            product *= row

        product = np.transpose(product)

        return np.array([product.mean(), product.std()])

    return columns_product


def crossing_rate_factory(type='zero'):
    def crossing_rate(array):
        if type is 'zero':
            if len(array.shape) > 1:
                means = np.zeros(array.shape[1])
            else:
                means = 0
        elif type is 'mean':
            means = np.average(array, axis=0)

        crossings = []
        for i in range(1, array.shape[0]):
            crossings.append(np.abs(np.sign(array[i] - means) - np.sign(array[i - 1] - means)))

        final_crossing_rate = np.sum(crossings, axis=0) / (array.shape[0] - 1)

        return final_crossing_rate

    return crossing_rate


def root_square_mean(array):
    squares = np.square(array)
    means = np.average(squares, axis=0)
    square_roots = np.sqrt(means)

    return square_roots


def energy(array):
    means = np.average(array, axis=0)
    calibrated_values = array - means
    squared = np.square(calibrated_values)
    axis_by_axis_energy = np.sqrt(np.average(squared, axis=0))

    average_energy = np.average(axis_by_axis_energy)

    return average_energy


def median(array):
    return np.median(array, axis=0)


def pearson_correlation(array):
    coefs = np.corrcoef(array, rowvar=0)

    results = []

    for i in range(array.shape[1]):
        for j in range(i + 1, array.shape[1]):
            j_ = coefs[i, j]

            if np.isnan(j_):
                j_ = 0

            results.append(j_)

    return np.array(results)


def skewness(array):
    # Taken from Wearable Mobility Monitoring Using a Multimedia Smartphone Platform by Hache et al
    return scipy.stats.skew(array, axis=0)


def maxmin_range(array):
    return np.max(array, axis=0) - np.min(array, axis=0)


def interquartile_range(array):
    q75, q25 = np.percentile(array, [75, 25], axis=0)
    return q75 - q25


def magnitude_avg_and_std(array):
    magnitude = np.linalg.norm(array, axis=1)
    return np.average(magnitude), np.std(magnitude)


def gravity_vector(array):
    g = np.zeros_like(array)

    for i in range(1, array.shape[0]):
        g[i] = 0.9 * g[i - 1] + 0.1 * array[i]

    return np.average(g, axis=0)


def frequency_domain_factory(sample_rate):
    def frequency_domain_features(array):
        fourier_transform = np.fft.rfft(array, axis=0)
        frequency_powers = np.abs(fourier_transform)

        means = np.mean(frequency_powers, axis=0)
        stds = np.std(frequency_powers, axis=0)

        max_power = np.max(frequency_powers, axis=0)
        median_power = np.median(frequency_powers, axis=0)

        sample_spacing = 1 / sample_rate
        frequencies = np.fft.rfftfreq(array.shape[0], sample_spacing)

        spectral_centroid = np.sum(frequency_powers * frequencies[:, np.newaxis], axis=0) / np.sum(frequency_powers,
                                                                                                   axis=0)

        for i in range(len(spectral_centroid)):
            if not np.isfinite(spectral_centroid[i]):
                spectral_centroid[i] = 0

        dominant_frequencies_indices = np.argmax(frequency_powers, axis=0)
        dominant_frequencies = np.array([frequencies[i] for i in dominant_frequencies_indices])

        frequency_power_squares = np.square(frequency_powers, np.zeros_like(frequency_powers)) / frequency_powers.shape[
            0]
        p_i = frequency_power_squares / np.sum(frequency_power_squares, axis=0)

        entropy = scipy.stats.entropy(p_i)

        for i in range(len(entropy)):
            if not np.isfinite(entropy[i]):
                entropy[i] = 0

        return np.hstack([means, stds, max_power, median_power, spectral_centroid, dominant_frequencies, entropy])

    return frequency_domain_features


class DataLoader:
    functions = {
        'means_and_std': [means_and_std_factory(False)],
        'abs_means_and_std': [means_and_std_factory(True)],
        'peak_acceleration': [peak_acceleration],
        'most_common': [most_frequent_value],
        'zero_crossing_rate': [crossing_rate_factory('zero')],
        'mean_crossing_rate': [crossing_rate_factory('mean')],
        'root_square_mean': [root_square_mean],
        'energy': [energy],
        'median': [median],
        'skewness': [skewness],
        'correlation': [pearson_correlation],
        'maxmin_range': [maxmin_range],
        'interquartile_range': [interquartile_range],
        'magnitude_avg_and_std': [magnitude_avg_and_std],
        'gravity_vector': [gravity_vector]
    }

    def __init__(self, sample_rate=100, window_length=2.0, degree_of_overlap=0.0):
        self.sample_rate = sample_rate
        self.window_samples = int(round(window_length * sample_rate))
        self.step_size = int(round((1 - degree_of_overlap) * self.window_samples))
        self.functions["frequency_features_v1"] = [frequency_domain_factory(self.sample_rate)]

    def read_data(self, file_path, func_keywords, abs_vals=False, dtype="float", relabel_dict=None):
        sensor_data = np.loadtxt(fname=file_path, delimiter=",", dtype=dtype)

        if relabel_dict:
            for k in relabel_dict:
                np.place(sensor_data, sensor_data == k, [relabel_dict[k]])

        if not func_keywords:
            return sensor_data

        functions = []

        if len(sensor_data.shape) > 1:
            column_index_combinations = generate_all_integer_combinations(sensor_data.shape[1])

            self.functions["column_products"] = [column_product_factory(t) for t in column_index_combinations]

        for name in sorted(func_keywords):
            functions += self.functions[name]

        all_features = []

        for window_start in range(0, len(sensor_data), self.step_size):
            window_end = window_start + self.window_samples
            if window_end > len(sensor_data):
                break
            window = sensor_data[window_start:window_end]

            extracted_features = [func(window) for func in functions]
            all_features.append(np.hstack(extracted_features))

        one_large_array = np.vstack(all_features)

        if abs_vals:
            np.absolute(one_large_array, one_large_array)

        return one_large_array

    def read_sensor_data(self, file_path, abs_vals=False):
        keywords = ["means_and_std", "abs_means_and_std", "peak_acceleration", "column_products", "skewness",
                    "zero_crossing_rate", "mean_crossing_rate", "root_square_mean", "energy", "median", "maxmin_range",
                    "interquartile_range", "magnitude_avg_and_std", "correlation", "frequency_features_v1"]

        return self.read_data(file_path, keywords, abs_vals=abs_vals)

    def read_label_data(self, file_path, relabel_dict):
        return self.read_data(file_path, ["most_common"], dtype="int", relabel_dict=relabel_dict).ravel()
