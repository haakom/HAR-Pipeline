from __future__ import print_function

import matplotlib
from collections import Counter

from hunt_dataset_definitions import label_to_number_dict

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np

from definitions import PROJECT_ROOT


def count_occurrences(path):
    occs = Counter()
    for root, _, files in os.walk(path):
        for f in files:
            if "labels" in f:
                print(f)
                this_labels = np.loadtxt(fname=os.path.join(root, f), delimiter=",", dtype="int")
                occs.update(this_labels)

    return occs


def text_plot(labels, values, title="", output_paths=None, show=False):
    if output_paths is None:
        print("No output paths specified")
        output_paths = []
    times_font = {'fontname': 'Times New Roman'}
    fig, ax = plt.subplots(figsize=(16.1803 / 2.5, 10 / 2.5))
    fig.set_dpi(300)

    def autolabel(rects):
        for i, rect in enumerate(rects):
            width = rect.get_width()
            ax.text(width,
                    i,
                    '%d' % int(width),
                    ha='right', va='center', fontsize=10, zorder=4000)

    ax.grid(b=True, which='major', linestyle='-', axis='x', zorder=0)
    ax.grid(b=True, which='minor', linestyle='dotted', axis='x', zorder=0)

    plt.xlim(10, 300000)
    rects1 = ax.barh(bottom=range(len(labels)), tick_label=labels, width=values, log=True, zorder=1000)

    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_zorder(2000)
    # ax.set_ylabel("Activity", **times_font)
    # ax.set_title(title, **times_font)
    # ax.set_xlabel("Samples", **times_font)

    autolabel(rects1)
    plt.tight_layout()
    for p in output_paths:
        plt.savefig(p)
    if show:
        plt.show()


def count_and_make_plot_for_subject(data_folder, output_paths=None, show=False):
    occurrences = count_occurrences(data_folder)

    for activity in [20, 21, 22, 23]:
        if activity in occurrences:
            occurrences[8] += occurrences[activity]
            occurrences.pop(activity)

    bar_chart_activities = ['undefined', 'transition', 'cycling (sit)', 'running', 'stairs (ascending)',
                            'stairs (descending)', 'walking', 'picking', 'bending', 'non-vigorous activity',
                            'shuffling',
                            'standing', 'sitting', 'lying']
    activities_as_numbers = [label_to_number_dict[a] for a in bar_chart_activities]

    bar_values = []

    for k in activities_as_numbers:
        if k in occurrences:
            bar_values.append(occurrences[k])
        else:
            bar_values.append(0)

    print("making plot")
    text_plot(bar_chart_activities, bar_values, output_paths=output_paths, show=show)


if __name__ == "__main__":
    subjects = ["S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15",
                "S16"]

    for subject in subjects:
        subject_folder = os.path.join(PROJECT_ROOT, "DATA", "stroke_patients", subject)
        eps_path = os.path.join(PROJECT_ROOT, "DATA", "activity_distribution_plots", subject + ".eps")
        png_path = os.path.join(PROJECT_ROOT, "DATA", "activity_distribution_plots", subject + ".png")
        count_and_make_plot_for_subject(subject_folder, [eps_path, png_path], show=False)

    print("Done with them all!")
