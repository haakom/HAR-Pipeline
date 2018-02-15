import keras
from keras.layers import Input, Dropout, Activation, Concatenate, Bidirectional, Dense, BatchNormalization
from keras.models import Model
import read_data as rd
import numpy as np
import sys
import tensorflow as tf

# Load training files
BATCH_SIZE = 128
NUM_ACTIVITIES = 19
NUM_FEATURES = 6
NUM_EPOCHS = 512
predict_sequences = False
SEQUENCE_LENGTH = 200
ALL_SUBJECTS = ["006", "008", "009", "010", "011", "012", "013",
                "014", "015", "016", "017", "018", "019",
                "020", "021", "022"]



evaluation_list = []  # List for storing evalutations for each subject in

for subject in ALL_SUBJECTS:

    print "Building network..."
    if "guest" not in sys.path[0]:
        from keras.layers import CuDNNLSTM

        # Build LSTM RNN GPU
        nn_in1 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn_in2 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn1 = Bidirectional(CuDNNLSTM(units=5, return_sequences=True))(nn_in1)
        nn2 = Bidirectional(CuDNNLSTM(units=5, return_sequences=True))(nn_in2)
        nn = Concatenate(axis=2)([nn1, nn2])
        nn = Bidirectional(CuDNNLSTM(units=10, return_sequences=predict_sequences))(nn)
        nn = Dropout(0.9)(nn)
        nn = Dense(NUM_ACTIVITIES)(nn)
        nn = Activation(activation="softmax")(nn)
    else:
        from keras.layers import LSTM

        # Build LSTM RNN CPU
        nn_in1 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn_in2 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn1 = Bidirectional(LSTM(units=5, return_sequences=True))(nn_in1)
        nn2 = Bidirectional(LSTM(units=5, return_sequences=True))(nn_in2)
        nn = Concatenate(axis=2)([nn1, nn2])
        nn = Bidirectional(LSTM(units=10, return_sequences=predict_sequences))(nn)
        nn = Dropout(0.9)(nn)
        nn = Dense(NUM_ACTIVITIES)(nn)
        nn = Activation(activation="softmax")(nn)

    model = Model(inputs=[nn_in1, nn_in2], outputs=nn)
    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])

    model.summary()
    # Automatic selection of dataset location depending on which machine the code is running on
    print "Reading data..."
    if "guest" not in sys.path[0]:
        train_x, train_y, val_x, val_y = rd.build_dataset(subject, "/lhome/haakom/HUNT_Project/HAR-Pipeline/DATA/combined_in_and_out_of_lab",
                                                          SEQUENCE_LENGTH, NUM_ACTIVITIES,
                                                          use_most_common_label=not predict_sequences,
                                                          print_stats=True,
                                                          normalize_data=True,
                                                          dataset=1)
    else:
        train_x, train_y, val_x, val_y  = rd.build_dataset(subject, "/home/guest/Documents/HAR-Pipeline/DATA/combined_in_and_out_of_lab",
                                                           SEQUENCE_LENGTH, NUM_ACTIVITIES,
                                                           use_most_common_label=not predict_sequences,
                                                           print_stats=False,
                                                           normalize_data=True,
                                                           dataset=1)





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


    patience_factor = 3
    train_history = model.fit([train_x1, train_x2], train_y,
                              epochs=NUM_EPOCHS,
                              batch_size=BATCH_SIZE,
                              validation_data=([val_x1, val_x2], val_y),
                              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                       min_delta=0.001,
                                                                       patience=patience_factor,
                                                                       verbose=1,
                                                                       mode='min')]) # Training
    print "Appending: " + str(train_history.history['val_acc'][-(patience_factor+1)]) + " to eval list"
    evaluation_list.append(train_history.history['val_acc'][-(patience_factor+1)])





print "Average accuracy: " + str(sum(evaluation_list)/float(len(evaluation_list)))
print "Evaluation List: " + str(evaluation_list)
