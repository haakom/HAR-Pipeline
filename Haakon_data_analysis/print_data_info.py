import read_datasets as rd
import numpy as np


train_x, train_y, = rd.read_dataset("/home/guest/Documents/HAR-Pipeline/DATA/trene", 1, 200)

print np.unique(train_y)

