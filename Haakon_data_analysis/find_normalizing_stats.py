from Haakon_Recurrent_ANN import read_data as rd
import numpy as np

train_x, train_y, val, val2 = rd.build_dataset(None, "/home/guest/Documents/Downsampled-data/Without_Outliers_Resample", 1, 19, use_most_common_label=False, print_stats=True, generate_one_hot=False, normalize_data=False)
train_x = np.reshape(train_x, newshape=[train_x.shape[0], train_x.shape[2]])
train_t = np.reshape(train_y, newshape=[train_y.shape[0], train_y.shape[2]])

mean_list = []
std_list = []
for i in range(train_x.shape[1]):
    mean = np.mean(train_x[:, i])
    std = np.std(train_x[:,i])
    mean_list.append(mean)
    std_list.append(std)


print mean_list
print std_list