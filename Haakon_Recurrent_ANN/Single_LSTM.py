import keras
from keras.layers import Input, Dropout, Activation, Concatenate
from keras.models import Model
import read_data as rd
import numpy as np
import sys
import tensorflow as tf

# Load training files
BATCH_SIZE = 1024
NUM_ACTIVITIES = 19
NUM_FEATURES = 6
NUM_EPOCHS = 64
predict_sequences = False
SEQUENCE_LENGTH = 100
ALL_SUBJECTS = ["006", "008", "010", "011", "012", "013",
                "014", "015", "016", "017", "018", "019",
                "020", "021", "022"]



evaluation_list = []  # List for storing evalutations for each subject in

print "Reading data..."
for subject in ALL_SUBJECTS:
    # Automatic selection of dataset location depending on which machine the code is running on
    if "guest" not in sys.path[0]:
        train_x, train_y, val_x, val_y = rd.build_dataset(subject,
                                                          "/lhome/haakom/HUNT_Project/HAR-Pipeline/DATA/combined_in_and_out_of_lab",
                                                          SEQUENCE_LENGTH, NUM_ACTIVITIES,
                                                          use_most_common_label=not predict_sequences,
                                                          print_stats=True,
                                                          normalize_data=True,
                                                          dataset=1)
    else:
        train_x, train_y, val_x, val_y = rd.build_dataset(subject,
                                                          "/home/guest/Documents/HAR-Pipeline/DATA/combined_in_and_out_of_lab",
                                                          SEQUENCE_LENGTH, NUM_ACTIVITIES,
                                                          use_most_common_label=not predict_sequences,
                                                          print_stats=False,
                                                          normalize_data=True,
                                                          dataset=1)

    print "Building network..."
    if "guest" not in sys.path[0]:
        from keras.layers import CuDNNLSTM

        # Build LSTM RNN GPU
        nn_in = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES])
        nn = CuDNNLSTM(units=6, return_sequences=True)(nn_in)
        nn = CuDNNLSTM(units=NUM_ACTIVITIES, return_sequences=predict_sequences)(nn)
        nn = Activation(activation="softmax")(nn)
    else:
        from keras.layers import LSTM

        # Build LSTM RNN CPU
        nn_in = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES])
        nn = LSTM(units=6, return_sequences=True)(nn_in)
        nn = LSTM(units=NUM_ACTIVITIES, return_sequences=predict_sequences)(nn)
        nn = Activation(activation="softmax")(nn)

    model = Model(inputs=nn_in, outputs=nn)

    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])

    model.summary()

    epochs_evaluation_list = []
    for epoch in range(NUM_EPOCHS):
        print "It's epoch " + str(epoch) + "/" + str(NUM_EPOCHS) + " gaddamit! Not 1/1!"
        model.fit(train_x, train_y, epochs=1, batch_size=BATCH_SIZE)
        loss, accuracy =  model.evaluate(val_x, val_y, batch_size=BATCH_SIZE, verbose=0)
        print "Validation accuracy: " + str(accuracy)
        epochs_evaluation_list.append(accuracy)

    evaluation_list.append(max(epochs_evaluation_list))




print "Average accuracy: " + str(sum(evaluation_list)/float(len(evaluation_list)))
print "Evaluation List: " + str(evaluation_list)
