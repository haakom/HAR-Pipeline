from data import get_data_set
from cnn import ConvolutionalNeuralNetwork
from viterbi import run_viterbi
import numpy as np
import TRAINING_VARIABLES
import pandas as pd
import matplotlib.pyplot as plt
import json

V = TRAINING_VARIABLES.VARS()


def main():
    # Load test data
    # Input: Testing, generate new windows, oversampling, viterbi training
    data_type = "predicting"
    generate_new_windows = True
    oversampling = False
    viterbi = False
    data_set = get_data_set(data_type, generate_new_windows, oversampling, viterbi, V.PREDICTING_PATH)

    # Create network
    cnn = ConvolutionalNeuralNetwork()
    cnn.set_data_set(data_set)
    cnn.load_model()

    cnn_result = cnn.get_predictions()

    viterbi_result = run_viterbi()

    data_frame = pd.DataFrame(viterbi_result)

    data_frame.to_csv(V.VITERBI_RESULT_PREDICTING)  # TODO: Results are the same every run. Source of error unknown.

    print 'Prediction saved at path', V.VITERBI_RESULT_PREDICTING


# TODO: This is a duplicate of a function in testing.py
def produce_statistics_json(result, save_path='RESULTS/TEST_STATISTICS.json'):
    score = get_score(result)

    specificity = {}
    precision = {}
    recall = {}
    for i in range(0, len(score[1])):
        specificity[V.ACTIVITY_NAMES_CONVERTION[i + 1]] = score[1][i]
        precision[V.ACTIVITY_NAMES_CONVERTION[i + 1]] = score[2][i]
        recall[V.ACTIVITY_NAMES_CONVERTION[i + 1]] = score[3][i]

    statistics = {
        'ACCURACY': score[0],
        'SPECIFICITY': specificity,
        'PRECISION': precision,
        'RECALL': recall
    }

    with open(save_path, "w") as outfile:
        json.dump(statistics, outfile)
    return statistics


def get_score(result_matrix):
    activities = V.ACTIVITIES
    '''TP / (FP - TP)
    Correctly classified walking / Classified as walking
    '''
    true_positives = np.zeros(len(activities))
    true_negatives = np.zeros(len(activities))

    fp_tp = np.zeros(len(activities))
    tp_fn = np.zeros(len(activities))
    fp_tn = np.zeros(len(activities))

    actual = result_matrix[:, 0]
    predicted = result_matrix[:, 2]

    for activity in activities:
        ''' FP - TP'''
        fp_tp[activity - 1] = np.sum(predicted == activity)  # len(df[df[0]==activity])
        ''' TP - FN '''
        tp_fn[activity - 1] = np.sum(actual == activity)  # len(df_actual[df_actual[0]==activity])
        ''' FP - TN '''
        fp_tn[activity - 1] = np.sum(actual != activity)  # len(df_actual[df_actual[0] != activity])

    for i in range(0, len(predicted)):
        if predicted[i] == actual[i]:
            true_positives[actual[i] - 1] += 1.0

        for activity in activities:
            if actual[i] != activity and predicted[i] != activity:
                true_negatives[activity - 1] += 1.0

    accuracy = sum(true_positives) / sum(tp_fn)
    specificity = true_negatives / fp_tn
    precision = true_positives / fp_tp
    recall = true_positives / tp_fn
    return [accuracy, specificity, precision, recall]


def visualize(result_matrix):
    for i in range(0, len(result_matrix)):
        result_matrix[i][0] = V.VISUALIZATION_CONVERTION[result_matrix[i][0] + 1]
        result_matrix[i][1] = V.VISUALIZATION_CONVERTION[result_matrix[i][1] + 1]
        result_matrix[i][2] = V.VISUALIZATION_CONVERTION[result_matrix[i][2] + 1]

    start = 0
    stop = 1000
    actual = result_matrix[:, 0][start:stop]
    cnn = result_matrix[:, 1][start:stop]
    viterbi = result_matrix[:, 2][start:stop]

    # t = cnn != viterbi
    # actual = actual[t]
    # cnn = cnn[t]
    # viterbi = viterbi[t]

    y_values = ["Lying", "Sit", "Stand", "Walk", "Walk(up)", "Walk(down)", "Cycle (sit)", "Cycle(Stand)", "Bending",
                "Running"]
    y_axis = np.arange(1, 11, 1)

    plt.figure(1)

    plt.subplot(311)
    axes = plt.gca()
    axes.set_ylim([0.9, 10.4])
    plt.yticks(y_axis, y_values)
    plt.plot(actual)

    plt.subplot(312)
    axes = plt.gca()
    axes.set_ylim([0.9, 10.4])
    plt.yticks(y_axis, y_values)
    plt.plot(cnn)

    plt.subplot(313)
    axes = plt.gca()
    axes.set_ylim([0.9, 10.4])
    plt.yticks(y_axis, y_values)
    plt.plot(viterbi)
    plt.show()


def show_confusion_matrix(result_matrix, index):
    for i in range(0, len(result_matrix)):
        result_matrix[i][0] = V.VISUALIZATION_CONVERTION[result_matrix[i][0] + 1]
        result_matrix[i][1] = V.VISUALIZATION_CONVERTION[result_matrix[i][1] + 1]
        result_matrix[i][2] = V.VISUALIZATION_CONVERTION[result_matrix[i][2] + 1]

    confusion_matrix = np.zeros((len(V.ACTIVITIES), len(V.ACTIVITIES)))
    for i in range(0, len(result_matrix)):
        actual = result_matrix[i][0]
        predicted = result_matrix[i][index]
        confusion_matrix[actual - 1][predicted - 1] += 1.0

    row_sums = confusion_matrix.sum(axis=1)
    norm_conf = confusion_matrix / row_sums[:, np.newaxis]

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.summer,
                    interpolation='nearest')

    width = len(confusion_matrix)
    height = len(confusion_matrix[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(confusion_matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)

    plt.title('Confusion Matrix')
    labels = ['Lying', 'Sitting', 'Standing', 'Walking', 'Stairs (up)', 'Stairs (down)', 'Cycling (sit)',
              'Cycling (stand)', 'Bending', 'Running']
    plt.xticks(range(width), labels, rotation='vertical')
    plt.yticks(range(height), labels)
    plt.show()


if __name__ == "__main__":
    main()
