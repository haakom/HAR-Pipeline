from Haakon_Recurrent_ANN import read_data as rd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score


# Load training files
skip_rate = 100
BATCH_SIZE = 512
NUM_ACTIVITIES = 20
NUM_FEATURES = 6
NUM_EPOCHS = 64
SEQUENCE_LENGTH = 1
ALL_SUBJECTS = ["006", "008", "010", "011", "012", "013",
                "014", "015", "016", "017", "018", "019",
                "020", "021", "022"]

evaluation_list = []  # List for storing evalutations for each subject in

for subject in ALL_SUBJECTS:
    # Automatic selection of dataset location depending on which machine the code is running on
    print "Reading data..."
    if "guest" not in sys.path[0]:
        train_x, train_y, val_x, val_y = rd.build_dataset(subject, "/lhome/haakom/HUNT_Project/HAR-Pipeline/DATA/combined_in_and_out_of_lab", SEQUENCE_LENGTH, NUM_ACTIVITIES, print_stats=True)
    else:
        train_x, train_y, val_x, val_y  = rd.build_dataset(subject, "/home/guest/Documents/HAR-Pipeline/DATA/trene", SEQUENCE_LENGTH, NUM_ACTIVITIES, use_most_common_label=False, print_stats=True, generate_one_hot=False)

    train_x = np.reshape(train_x, newshape=[train_x.shape[0], train_x.shape[2]])
    train_y = np.reshape(train_y, newshape=[train_y.shape[0]])

    val_x = np.reshape(val_x, newshape=[val_x.shape[0], val_x.shape[2]])
    val_y = np.reshape(val_y, newshape=[val_y.shape[0]])
    print train_x.shape
    print train_y.shape
    """
    # Separate thigh and back training data into separate "channels"
    train_x1= np.zeros(shape=[train_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES/2])
    train_x2= np.zeros(shape=[train_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES/2])

    val_x1 = np.zeros(shape=[val_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES / 2])
    val_x2 = np.zeros(shape=[val_x.shape[0], SEQUENCE_LENGTH, NUM_FEATURES / 2])

    for example in range(train_x.shape[0]):
        for i in range(SEQUENCE_LENGTH):
            train_x1[example, i, :] = train_x[example, i, 0:3]
            train_x2[example, i, :] = train_x[example, i, 3:6]

    for example in range(val_x.shape[0]):
        for i in range(SEQUENCE_LENGTH):
            val_x1[example, i, :] = val_x[example, i, 0:3]
            val_x2[example, i, :] = val_x[example, i, 3:6]
    """

    print "Training forest"
    overall_forest = RFC(n_estimators=50, class_weight="balanced", n_jobs=-1)
    overall_forest.fit(train_x[::skip_rate], train_y[::skip_rate])
    predictions = overall_forest.predict(val_x)

    correct=0
    false=0
    print "Train Accuracy :: ", accuracy_score(train_y[::skip_rate], overall_forest.predict(train_x[::skip_rate]))
    print "Val Accuracy  :: ", accuracy_score(val_y, predictions)
    evaluation_list.append(accuracy_score(val_y, predictions))

print "Average validation accuracy :: " + str(sum(evaluation_list)/float(len(evaluation_list)))
