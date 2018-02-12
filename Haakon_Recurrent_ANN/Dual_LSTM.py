import keras
from keras.layers import Input, Dropout, Activation, Concatenate
from keras.models import Model
import read_data as rd
import numpy as np
import sys
import tensorflow as tf

# Load training files
BATCH_SIZE = 4096
NUM_ACTIVITIES = 20
NUM_FEATURES = 6
NUM_EPOCHS = 64
use_most_common_label_in_sequence = True
SEQUENCE_LENGTH = 500
ALL_SUBJECTS = ["006", "008", "010", "011", "012", "013",
                "014", "015", "016", "017", "018", "019",
                "020", "021", "022"]

print not use_most_common_label_in_sequence


evaluation_list = []  # List for storing evalutations for each subject in

for subject in ALL_SUBJECTS:

    print "Building network..."
    if "guest" not in sys.path[0]:
        from keras.layers import CuDNNLSTM

        # Build LSTM RNN GPU
        nn_in1 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn_in2 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn1 = CuDNNLSTM(units=3, return_sequences=True)(nn_in1)
        nn2 = CuDNNLSTM(units=3, return_sequences=True)(nn_in2)
        nn = Concatenate(axis=2)([nn1, nn2])
        nn = CuDNNLSTM(units=NUM_ACTIVITIES, return_sequences=not use_most_common_label_in_sequence)(nn)
        nn = Activation(activation="softmax")(nn)
    else:
        from keras.layers import LSTM

        # Build LSTM RNN CPU
        nn_in1 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn_in2 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn1 = LSTM(units=3, return_sequences=True)(nn_in1)
        nn2 = LSTM(units=3, return_sequences=True)(nn_in2)
        nn = Concatenate(axis=2)([nn1, nn2])
        nn = LSTM(units=NUM_ACTIVITIES, return_sequences=not use_most_common_label_in_sequence)(nn)
        nn = Activation(activation="softmax")(nn)

    model = Model(inputs=[nn_in1, nn_in2], outputs=nn)
    adgrad_optimizer = keras.optimizers.adagrad(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=adgrad_optimizer, metrics=["accuracy"])

    model.summary()
    # Automatic selection of dataset location depending on which machine the code is running on
    print "Reading data..."
    if "guest" not in sys.path[0]:
        train_x, train_y, val_x, val_y = rd.build_dataset(subject, "/lhome/haakom/HUNT_Project/HAR-Pipeline/DATA/combined_in_and_out_of_lab", SEQUENCE_LENGTH, NUM_ACTIVITIES, use_most_common_label=use_most_common_label_in_sequence, print_stats=True)
    else:
        train_x, train_y, val_x, val_y  = rd.build_dataset(subject, "/home/guest/Documents/HAR-Pipeline/DATA/combined_in_and_out_of_lab", SEQUENCE_LENGTH, NUM_ACTIVITIES, use_most_common_label=use_most_common_label_in_sequence, print_stats=True)

    print train_x.shape
    print train_y.shape

    print val_x.shape
    print val_y.shape


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

    epochs_evaluation_list = []

    for epoch in range(NUM_EPOCHS):
        print "It's epoch " + str(epoch+1) + "/" + str(NUM_EPOCHS) + " gaddamit! Not 1/1!"
        model.fit([train_x1, train_x2], train_y, epochs=1, batch_size=BATCH_SIZE) # Training
        loss, accuracy =  model.evaluate([val_x1, val_x2], val_y, batch_size=BATCH_SIZE, verbose=0) # Validation
        print model.predict([val_x1, val_x2])[0]
        print val_y[0]
        exit()
        print "Validation accuracy: " + str(accuracy)
        epochs_evaluation_list.append(accuracy)

    evaluation_list.append(max(epochs_evaluation_list))




print "Average accuracy: " + str(sum(evaluation_list)/float(len(evaluation_list)))
print "Evaluation List: " + str(evaluation_list)
