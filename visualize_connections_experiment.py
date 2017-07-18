from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import json
import os

from definitions import PROJECT_ROOT
from hunt_dataset_definitions import user_tug_walk_results, label_to_number_dict


def scatter_plot_with_arrows(xs, ys, colors, xlabel, ylabel, annotation_coordinates, annotation_texts,
                             arrow_starts=None, arrow_ends=None, cmap_name="copper", title=""):
    if arrow_starts is None:
        arrow_starts = []

    if arrow_ends is None:
        arrow_ends = []

    times_font = {'fontname': 'Times New Roman'}

    # Taken from https://stackoverflow.com/questions/9127434/how-to-create-major-and-minor-gridlines-with-different-linestyles-in-python
    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(xs, ys, c=colors, s=20, cmap=plt.get_cmap(cmap_name))
    ax.set_title(title)
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # plt.ylim(0, 30)
    # plt.xlim(0, 30)
    plt.xlabel(xlabel, **times_font)
    plt.ylabel(ylabel, **times_font)

    ax.set_xticklabels(ax.get_xticks(), times_font)
    ax.set_yticklabels(ax.get_yticks(), times_font)
    fig.set_dpi(200)

    # Taken from arrow example: http://matplotlib.org/users/annotations_intro.html
    for start, end in zip(arrow_starts, arrow_ends):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(facecolor='black', shrink=0, width=.01, headwidth=5, headlength=5))

    for coordinate, s in zip(annotation_coordinates, annotation_texts):
        x, y = coordinate
        ax.text(x, y, s, fontsize=8, **times_font)

    plt.show()


def visualize_multi_personalization_file(path, activity, title=""):
    with open(path, "r") as f:
        best_data = json.load(f)

    starts, ends = [], []

    for k in best_data:
        starts.append(k)
        ends.append(best_data[k][activity])

    visualize_subject_connections(starts, ends, show_healthy=False, cmap_name="bwr", title=title)


def visualize_single_personalization_file(path, title=""):
    with open(path, "r") as f:
        best_data = json.load(f)

    starts, ends = [], []

    for k in best_data:
        starts.append(k)
        ends.append(best_data[k])

    visualize_subject_connections(starts, ends, show_healthy=False, cmap_name="bwr", title=title)


def visualize_subject_connections(start_subject_ids, end_subject_ids, x_data_column="walk", y_data_column="tug",
                                  show_healthy=True, cmap_name="copper", title=""):
    subject_ids = sorted(list(user_tug_walk_results.keys()))
    if not show_healthy:
        subject_ids.remove("H")
    color_dict = {None: 0.5, "M": 0, "F": 1}

    x_data = [user_tug_walk_results[k][x_data_column] for k in subject_ids]
    y_data = [user_tug_walk_results[k][y_data_column] for k in subject_ids]
    subject_colors = [color_dict[user_tug_walk_results[k]["gender"]] for k in subject_ids]

    subject_coordinates = dict((i, (w, t)) for i, w, t in zip(subject_ids, x_data, y_data))

    id_annotation_coordinates = []
    for s in subject_ids:
        if s not in ["S13", "S14"]:
            c = np.array(subject_coordinates[s]) + 0.3 * np.array([1, 1])
        else:
            c = np.array(subject_coordinates[s]) + 0.3 * np.array([-5, 1])

        id_annotation_coordinates.append(c)

    a_starts = []
    a_ends = []

    for s_id, e_id in zip(start_subject_ids, end_subject_ids):
        s, e = subject_coordinates[s_id], subject_coordinates[e_id]
        a_starts.append(s)
        a_ends.append(e)

    scatter_plot_with_arrows(x_data, y_data, subject_colors, x_data_column, y_data_column,
                             id_annotation_coordinates,
                             subject_ids, a_starts, a_ends, cmap_name=cmap_name, title=title)


if __name__ == "__main__":
    relative_path = "VagesHAR/single_personalization/20170531_04_05_2_sensors_no_stairs/lt_rt/best_individual/all.json"
    title = os.path.basename(os.path.split(os.path.split(relative_path)[0])[0])
    visualize_single_personalization_file(os.path.join(PROJECT_ROOT, relative_path), title=title)

    """

    relative_path = "VagesHAR/multi_personalization/20170620_06_28_3_sensors_affected/aw_at_ut/adaptation/all.json"
    title = os.path.basename(os.path.split(os.path.split(relative_path)[0])[0])
    visualize_multi_personalization_file(os.path.join(PROJECT_ROOT, relative_path), str(label_to_number_dict["cycling (sit)"]), title=title)
    """