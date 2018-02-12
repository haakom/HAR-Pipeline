import keras
from keras.layers import Input, Dropout, Activation, Concatenate
from keras.models import Model
import read_data as rd
import numpy as np
import sys
import tensorflow as tf

# Load training files
BATCH_SIZE = 512
NUM_ACTIVITIES = 20
NUM_FEATURES = 6
NUM_EPOCHS = 64
SEQUENCE_LENGTH = 100
ALL_SUBJECTS = ["006", "008", "010", "011", "012", "013",
                "014", "015", "016", "017", "018", "019",
                "020", "021", "022"]


print "Building network..."
if "guest" not in sys.path[0]:
    from keras.layers import CuDNNLSTM
    # Build LSTM RNN GPU
    nn_in = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES])
    nn = CuDNNLSTM(units=3, return_sequences=True)(nn_in)
    nn = CuDNNLSTM(units=20, return_sequences=False)(nn)
    nn = Activation(activation="softmax")(nn)
else:
    from keras.layers import LSTM
    # Build LSTM RNN CPU
    nn_in = Input(shape=[SEQUENCE_LENGTH, NUM_FEATURES])
    nn = LSTM(units=6, return_sequences=True)(nn_in)
    nn = LSTM(units=20, return_sequences=False)(nn)
    nn = Activation(activation="softmax")(nn)

model = Model(inputs=nn_in, outputs=nn)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print "Reading data..."

evaluation_list = []  # List for storing evalutations for each subject in

for subject in ALL_SUBJECTS:
    # Automatic selection of dataset location depending on which machine the code is running on
    if "guest" not in sys.path[0]:
        train_x, train_y, val_x, val_y = rd.build_dataset(subject, "/lhome/haakom/HUNT_Project/HAR-Pipeline/DATA/all_files", SEQUENCE_LENGTH, NUM_ACTIVITIES)
    else:
        train_x, train_y, val_x, val_y  = rd.build_dataset(subject, "/home/guest/Documents/HAR-Pipeline/DATA/trene", SEQUENCE_LENGTH, NUM_ACTIVITIES)



    epochs_evaluation_list = []
    for epoch in range(NUM_EPOCHS):
        print "It's epoch " + str(epoch) + "/" + str(NUM_EPOCHS) + " gaddamit! Not 1/1!"
        model.fit(train_x, train_y, epochs=1, batch_size=BATCH_SIZE)
        print "Validating"
        loss, accuracy =  model.evaluate(val_x, val_y, batch_size=BATCH_SIZE)
        epochs_evaluation_list.append(accuracy)

    evaluation_list.append(max(epochs_evaluation_list))




print "Average accuracy: " + str(sum(evaluation_list)/float(len(evaluation_list)))
print "Evaluation List: " + str(evaluation_list)
