from __future__ import print_function, division

from collections import Counter
from datetime import datetime

import numpy as np
import os
import configparser

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.model_selection import train_test_split

from VagesHAR import VAGESHAR_ROOT
from VagesHAR.DataLoader import DataLoader as Dl
from VagesHAR.ClassifierPool import ClassifierPool
from VagesHAR.StatisticHelpers import generate_and_save_confusion_matrix, generate_and_save_statistics_json
from definitions import PROJECT_ROOT

label_to_number_dict = {
    "none": 0,
    "walking": 1,
    "running": 2,
    "shuffling": 3,
    "stairs (ascending)": 4,
    "stairs (descending)": 5,
    "standing": 6,
    "sitting": 7,
    "lying": 8,
    "transition": 9,
    "bending": 10,
    "picking": 11,
    "undefined": 12,
    "cycling (sit)": 13,
    "cycling (stand)": 14,
    "heel drop": 15,
    "vigorous activity": 16,
    "non-vigorous activity": 17,
    "Transport(sitting)": 18,
    "Commute(standing)": 19,
    "lying (prone)": 20,
    "lying (supine)": 21,
    "lying (left)": 22,
    "lying (right)": 23
}

number_to_label_dict = dict([(label_to_number_dict[l], l) for l in label_to_number_dict])


def neighbor_smooth_array(array):
    for i in range(1, len(array) - 1):
        previous_element = array[i - 1]
        next_element = array[i + 1]
        current_element = array[i]
        if previous_element != current_element and previous_element == next_element:
            array[i] = previous_element

    return array


def get_eligible_subjects(data_set_root, codeword_list, ignored_subjects=None):
    if ignored_subjects is None:
        ignored_subjects = []

    all_subjects_sensor_paths = dict()
    all_subjects_label_paths = dict()

    for root, _, files in os.walk(data_set_root):
        s_id = os.path.split(root)[1]
        if s_id in ignored_subjects:
            continue

        this_subjects_sensor_paths = []

        found_all_ks = True

        for k in codeword_list:
            found_k = False
            for f in files:
                if k in f:
                    this_subjects_sensor_paths.append(os.path.join(root, f))
                    found_k = True
                    break
            if not found_k:
                found_all_ks = False
                break

        if found_all_ks:
            all_subjects_sensor_paths[s_id] = this_subjects_sensor_paths
            all_subjects_label_paths[s_id] = os.path.join(root, s_id + "_labels.csv")

    return all_subjects_sensor_paths, all_subjects_label_paths


def load_sensors(csv_files, data_loader):
    read_data = [data_loader.read_sensor_data(f, abs_vals=ABSOLUTE_VALUES) for f in csv_files]
    if len(read_data) == 1:
        return read_data[0]
    return np.hstack(read_data)


def load_labels(csv_file, data_loader):
    return data_loader.read_label_data(csv_file, RELABEL_DICT)


def remove_unwanted_activities(sensor_data, label_data, keep_set):
    indices_to_keep = [i for i, a in enumerate(label_data) if a in keep_set]

    sensor_data = sensor_data[indices_to_keep]
    label_data = label_data[indices_to_keep]
    return sensor_data, label_data


def remove_singletons(sensor_data, label_data):
    indices_to_keep = list(range(len(label_data)))

    if remove_singletons:
        c = Counter(label_data)
        data_ = list(label_data)
        for k in c:
            if c[k] == 1:
                indices_to_keep.remove(data_.index(k))

    sensor_data = sensor_data[indices_to_keep]
    label_data = label_data[indices_to_keep]
    return sensor_data, label_data


def get_and_make_subdirectories(sub_name, *dirs):
    new_dirs = [os.path.join(d, sub_name) for d in dirs]
    map(lambda x: os.makedirs(x) if not os.path.exists(x) else None, new_dirs)
    return new_dirs


def run_experiments(experiment_collection_path, experiments=None):
    global RELABEL_DICT, ABSOLUTE_VALUES

    config = configparser.ConfigParser()

    collection_name = os.path.splitext(os.path.basename(experiment_collection_path))[0]
    print(experiment_collection_path)

    config.read_file(open(experiment_collection_path))
    now = datetime.now()
    datetime_prefix = now.strftime("%Y%m%d_%H_%M")
    all_experiments_id = datetime_prefix + "_" + collection_name
    top_statistics_folder = os.path.join(VAGESHAR_ROOT, "statistics", all_experiments_id)
    top_images_folder = os.path.join(VAGESHAR_ROOT, "images", all_experiments_id)

    if experiments is None:
        experiments = set(config.keys())
        experiments.discard("DEFAULT")

    for experiment_name in experiments:
        print(experiment_name)
        experiment_config = config[experiment_name]

        seed = experiment_config.getint("seed")

        activities_to_keep = {int(s) for s in experiment_config.get("activities_to_keep").split(", ")}
        RELABEL_DICT = dict({(int(k), int(v)) for k, v in
                             [s.split(", ") for s in experiment_config.get("relabel_dict").split("; ")]})

        n_trees = experiment_config.getint("n_trees", 50)
        max_depth = experiment_config.getint("max_depth", None)
        n_jobs = -1
        ABSOLUTE_VALUES = experiment_config.getboolean("absolute_values")

        run_adaptation_test = experiment_config.getboolean("adaptation_test")
        run_best_individual_test = experiment_config.getboolean("best_individual_test")
        run_ordinary_rfc_test = experiment_config.getboolean("global_model_test")
        use_neighbor_smoothing = experiment_config.getboolean("neighbor_smoothing")

        config_train_data_sets = experiment_config.get("train_data_sets").split(", ")
        config_train_sensor_codewords = experiment_config.get("train_sensor_codewords").split("; ")
        config_train_sensor_codewords = [s.split(", ") for s in config_train_sensor_codewords]
        config_test_data_sets = experiment_config.get("test_data_sets").split(", ")
        config_test_sensor_codewords = experiment_config.get("test_sensor_codewords").split("; ")
        config_test_sensor_codewords = [s.split(", ") for s in config_test_sensor_codewords]

        test_statistics_folder, test_images_folder = get_and_make_subdirectories(experiment_name,
                                                                                 top_statistics_folder,
                                                                                 top_images_folder)

        sample_frequency = experiment_config.getfloat("sample_frequency")
        window_length = experiment_config.getfloat("window_length")
        train_overlap = experiment_config.getfloat("train_overlap")
        test_overlap = experiment_config.getfloat("test_overlap")

        train_data_loader = Dl(sample_frequency=sample_frequency, window_length=window_length,
                               degree_of_overlap=train_overlap)
        test_data_loader = Dl(sample_frequency=sample_frequency, window_length=window_length,
                              degree_of_overlap=test_overlap)

        train_sensor_paths, train_label_paths = dict(), dict()
        test_sensor_paths, test_label_paths = dict(), dict()

        for s, c in zip(config_train_data_sets, config_train_sensor_codewords):
            temp_sensor_paths, temp_label_paths = get_eligible_subjects(
                data_set_root=os.path.join(PROJECT_ROOT, "DATA", s),
                codeword_list=c
            )
            train_sensor_paths.update(temp_sensor_paths)
            train_label_paths.update(temp_label_paths)

        for s, c in zip(config_test_data_sets, config_test_sensor_codewords):
            temp_sensor_paths, temp_label_paths = get_eligible_subjects(
                data_set_root=os.path.join(PROJECT_ROOT, "DATA", s),
                codeword_list=c
            )
            test_sensor_paths.update(temp_sensor_paths)
            test_label_paths.update(temp_label_paths)

        train_subject_ids = sorted(train_label_paths.keys())
        print("Training subjects:", train_subject_ids)
        test_subject_ids = sorted(test_label_paths.keys())
        print("Test subjects:", test_subject_ids)

        print("Loading training set")
        train_sensors = Parallel(n_jobs=n_jobs)(
            delayed(load_sensors)(train_sensor_paths[s_id], train_data_loader) for s_id in train_subject_ids)
        train_labels = Parallel(n_jobs=n_jobs)(
            delayed(load_labels)(train_label_paths[s_id], train_data_loader) for s_id in train_subject_ids)

        print("Loading test set")
        test_sensors = Parallel(n_jobs=n_jobs)(
            delayed(load_sensors)(test_sensor_paths[s_id], test_data_loader) for s_id in test_subject_ids)
        test_labels = Parallel(n_jobs=n_jobs)(
            delayed(load_labels)(test_label_paths[s_id], test_data_loader) for s_id in test_subject_ids)

        # Turn these into dictionaries
        train_sensor_dict = dict(
            [(s_id, sensor_windows) for s_id, sensor_windows in zip(train_subject_ids, train_sensors)])
        train_label_dict = dict(
            [(s_id, label_windows) for s_id, label_windows in zip(train_subject_ids, train_labels)])

        test_sensor_dict = dict(
            [(s_id, sensor_windows) for s_id, sensor_windows in zip(test_subject_ids, test_sensors)])
        test_label_dict = dict(
            [(s_id, label_windows) for s_id, label_windows in zip(test_subject_ids, test_labels)])

        # Cleaning up
        for s_id in train_subject_ids:
            sensors, labels = train_sensor_dict[s_id], train_label_dict[s_id]
            train_sensor_dict[s_id], train_label_dict[s_id] = remove_unwanted_activities(sensors, labels,
                                                                                         activities_to_keep)

        for s_id in test_subject_ids:
            sensors, labels = test_sensor_dict[s_id], test_label_dict[s_id]
            test_sensor_dict[s_id], test_label_dict[s_id] = remove_unwanted_activities(sensors, labels,
                                                                                       activities_to_keep)

        cp = ClassifierPool(classifier_type="RandomForest", random_state=seed, class_weight="balanced",
                            n_estimators=n_trees,
                            max_depth=max_depth, n_jobs=n_jobs)

        if run_adaptation_test or run_best_individual_test:
            print("Creating pool")
            for subject_id in sorted(train_subject_ids):
                print(subject_id)
                tmp_subject_sensors, tmp_subject_labels = train_sensor_dict[subject_id], train_label_dict[
                    subject_id]
                cp.train_subject_classifier(subject_id, tmp_subject_sensors, tmp_subject_labels)

        if run_adaptation_test:
            sub_test_name = "adaptation"
            print(sub_test_name)
            sub_test_statistics, sub_test_images = get_and_make_subdirectories(sub_test_name,
                                                                               test_statistics_folder,
                                                                               test_images_folder)

            overall_y_test, overall_y_pred = [], []

            for subject_id in test_subject_ids:
                print(subject_id)
                tmp_subject_sensors, tmp_subject_labels = test_sensor_dict[subject_id], test_label_dict[subject_id]
                tmp_subject_sensors, tmp_subject_labels = remove_singletons(tmp_subject_sensors, tmp_subject_labels)
                x_train, x_test, y_train, y_test = train_test_split(
                    tmp_subject_sensors, tmp_subject_labels, test_size=0.8, random_state=seed,
                    stratify=tmp_subject_labels)

                adapted, activity_to_subject_dict = cp.mix_new_classifier_from_pool(x_train, y_train, [subject_id])
                print(activity_to_subject_dict)
                y_pred = adapted.predict(x_test)

                if use_neighbor_smoothing:
                    y_pred = neighbor_smooth_array(y_pred)

                overall_y_test.append(y_test)
                overall_y_pred.append(y_pred)

                title = sub_test_name + " " + subject_id
                png_path = os.path.join(sub_test_images, subject_id + ".png")
                json_path = os.path.join(sub_test_statistics, subject_id + ".json")
                generate_and_save_confusion_matrix(y_test, y_pred, number_to_label_dict, png_path, title=title)
                generate_and_save_statistics_json(y_test, y_pred, number_to_label_dict, json_path)

            overall_y_test, overall_y_pred = np.hstack(overall_y_test), np.hstack(overall_y_pred)
            subject_id = "overall"

            title = sub_test_name + " " + subject_id
            png_path = os.path.join(sub_test_images, subject_id + ".png")
            json_path = os.path.join(sub_test_statistics, subject_id + ".json")
            generate_and_save_confusion_matrix(overall_y_test, overall_y_pred, number_to_label_dict, png_path,
                                               title=title)
            generate_and_save_statistics_json(overall_y_test, overall_y_pred, number_to_label_dict, json_path)

        if run_best_individual_test:
            sub_test_name = "best_individual"
            print(sub_test_name)
            sub_test_statistics, sub_test_images = get_and_make_subdirectories(sub_test_name,
                                                                               test_statistics_folder,
                                                                               test_images_folder)

            overall_y_test, overall_y_pred = [], []

            for subject_id in test_subject_ids:
                print(subject_id)

                tmp_subject_sensors, tmp_subject_labels = test_sensor_dict[subject_id], test_label_dict[subject_id]
                tmp_subject_sensors, tmp_subject_labels = remove_singletons(tmp_subject_sensors, tmp_subject_labels)

                x_train, x_test, y_train, y_test = train_test_split(
                    tmp_subject_sensors, tmp_subject_labels, test_size=0.8, random_state=seed,
                    stratify=tmp_subject_labels)
                best, best_subject_id = cp.find_best_existing_classifier(x_train, y_train, [subject_id])
                print(best_subject_id)
                y_pred = best.predict(x_test)

                if use_neighbor_smoothing:
                    y_pred = neighbor_smooth_array(y_pred)

                overall_y_test.append(y_test)
                overall_y_pred.append(y_pred)

                title = sub_test_name + " " + subject_id
                png_path = os.path.join(sub_test_images, subject_id + ".png")
                json_path = os.path.join(sub_test_statistics, subject_id + ".json")
                generate_and_save_confusion_matrix(y_test, y_pred, number_to_label_dict, png_path, title=title)
                generate_and_save_statistics_json(y_test, y_pred, number_to_label_dict, json_path)

            overall_y_test, overall_y_pred = np.hstack(overall_y_test), np.hstack(overall_y_pred)
            subject_id = "overall"

            title = sub_test_name + " " + subject_id
            png_path = os.path.join(sub_test_images, subject_id + ".png")
            json_path = os.path.join(sub_test_statistics, subject_id + ".json")
            generate_and_save_confusion_matrix(overall_y_test, overall_y_pred, number_to_label_dict, png_path,
                                               title=title)
            generate_and_save_statistics_json(overall_y_test, overall_y_pred, number_to_label_dict, json_path)

        if run_ordinary_rfc_test:
            sub_test_name = "general_population"
            print(sub_test_name)
            sub_test_statistics, sub_test_images = get_and_make_subdirectories(sub_test_name,
                                                                               test_statistics_folder,
                                                                               test_images_folder)

            overall_y_test, overall_y_pred = [], []

            for subject_id in test_subject_ids:
                png_path = os.path.join(sub_test_images, subject_id + ".png")
                json_path = os.path.join(sub_test_statistics, subject_id + ".json")
                print(subject_id)
                global_pool_ids = sorted(list(set(train_subject_ids) - {subject_id}))
                x_train = np.vstack([train_sensor_dict[x] for x in global_pool_ids])
                y_train = np.hstack([train_label_dict[x] for x in global_pool_ids])
                my_forest = Rfc(n_estimators=n_trees, random_state=seed, max_depth=max_depth, n_jobs=n_jobs,
                                class_weight="balanced")
                my_forest.fit(x_train, y_train)
                print("Testing")
                x_test = test_sensor_dict[subject_id]
                y_test = test_label_dict[subject_id]

                y_pred = my_forest.predict(x_test)

                if use_neighbor_smoothing:
                    y_pred = neighbor_smooth_array(y_pred)

                overall_y_test.append(y_test)
                overall_y_pred.append(y_pred)

                title = sub_test_name + " " + subject_id
                generate_and_save_confusion_matrix(y_test, y_pred, number_to_label_dict, png_path, title=title)
                generate_and_save_statistics_json(y_test, y_pred, number_to_label_dict, json_path)

            overall_y_test, overall_y_pred = np.hstack(overall_y_test), np.hstack(overall_y_pred)
            subject_id = "overall"

            title = sub_test_name + " " + subject_id
            png_path = os.path.join(sub_test_images, subject_id + ".png")
            json_path = os.path.join(sub_test_statistics, subject_id + ".json")
            generate_and_save_confusion_matrix(overall_y_test, overall_y_pred, number_to_label_dict, png_path,
                                               title=title)
            generate_and_save_statistics_json(overall_y_test, overall_y_pred, number_to_label_dict, json_path)


if __name__ == "__main__":
    run_experiments(os.path.join(VAGESHAR_ROOT, "configs", "train_healthy_test_stroke.cfg"))
