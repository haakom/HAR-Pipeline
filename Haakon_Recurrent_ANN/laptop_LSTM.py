import keras
from keras.layers import Input, Activation, Concatenate, BatchNormalization, Dropout, Bidirectional, Dense
from keras.models import Model
import read_data as rd
import numpy as np
import sys
import tensorflow as tf

# Load training files
BATCH_SIZE = 1024
NUM_ACTIVITIES = 19
NUM_FEATURES = 6
NUM_EPOCHS = 10
predict_sequences = False
SEQUENCE_LENGTH = 100
normalize_data = True
ALL_SUBJECTS = ["006", "008", "010", "011", "012", "013",
                "014", "015", "016", "017", "018", "019",
                "020", "021", "022"]



evaluation_list = []  # List for storing evalutations for each subject in

for subject in ALL_SUBJECTS:
    print "Building network..."
    if "guest" not in sys.path[0]:
        from keras.layers import CuDNNLSTM, CuDNNGRU

        # Build LSTM RNN GPU
        nn_in1 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn_in2 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn_in1 = BatchNormalization()(nn_in1)
        nn_in2 = BatchNormalization()(nn_in2)
        nn1 = CuDNNLSTM(units=3, return_sequences=True)(nn_in1)
        nn2 = CuDNNLSTM(units=3, return_sequences=True)(nn_in2)
        nn = Concatenate(axis=2)([nn1, nn2])
        nn = BatchNormalization()(nn)
        nn = CuDNNLSTM(units=NUM_ACTIVITIES, return_sequences=predict_sequences)(nn)
        nn = Activation(activation="softmax")(nn)
    else:
        from keras.layers import LSTM, GRU

        # Build LSTM RNN CPU
        nn_in1 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        nn_in2 = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES / 2])
        #nn_in1 = BatchNormalization()(nn_in1)
        #nn_in2 = BatchNormalization()(nn_in2)
        nn1 = Bidirectional(LSTM(units=5, return_sequences=False))(nn_in1)
        nn2 = Bidirectional(LSTM(units=5, return_sequences=False))(nn_in2)

        #nn1 = LSTM(units=5, return_sequences=True)(nn1)
        #nn2 = LSTM(units=5, return_sequences=True)(nn2)

        nn = Concatenate(axis=1)([nn1, nn2])
       #nn = BatchNormalization()(nn)
        #nn = Dropout(0.5)(nn)
        #nn = Bidirectional(LSTM(units=10, return_sequences=True))(nn)
        #nn = Bidirectional(LSTM(units=10, return_sequences=predict_sequences,recurrent_dropout=0.5))(nn)
        nn = Dropout(0.5)(nn)
        nn = Dense(NUM_ACTIVITIES)(nn)
        #nn = BatchNormalization()(nn)
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
                                                          normalize_data=normalize_data,
                                                          dataset=1)
    else:
        train_x, train_y, val_x, val_y  = rd.build_dataset(subject, "/home/guest/Documents/HAR-Pipeline/DATA/trene",
                                                           SEQUENCE_LENGTH, NUM_ACTIVITIES,
                                                           use_most_common_label=not predict_sequences,
                                                           print_stats=False,
                                                           normalize_data=normalize_data,
                                                           dataset=1)

    print train_x.shape
    train_x = train_x[0::10]
    train_y = train_y[0::10]

    print train_x.shape

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

    history = model.fit([train_x1, train_x2], train_y,
              epochs=10,
              batch_size=BATCH_SIZE,
              validation_data=([val_x1, val_x2], val_y),
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_acc',
                                                       min_delta=0.5,
                                                       patience=1,
                                                       verbose=1,
                                                       mode='max')])  # Training
    print history.history
    print history.history['val_acc'][-2]
    exit()
    loss, accuracy = model.evaluate([val_x1, val_x2], val_y, batch_size=BATCH_SIZE, verbose=0)  # Validation
    print "Validation accuracy: " + str(accuracy)
    epochs_evaluation_list.append(accuracy)

    evaluation_list.append(max(epochs_evaluation_list))




print "Average accuracy: " + str(sum(evaluation_list)/float(len(evaluation_list)))
print "Evaluation List: " + str(evaluation_list)
