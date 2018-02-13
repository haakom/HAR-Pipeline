import read_datasets as rd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab


stroke_x, stroke_y = rd.read_dataset("/home/guest/Documents/dataset", is_stroke=True)
stroke_mean_thigh = np.mean(stroke_x[:, 0:2])
stroke_std_thigh = np.sqrt(np.var(stroke_x[:, 0:2]))

stroke_mean_back = np.mean(stroke_x[:, 3:5])
stroke_std_back = np.sqrt(np.var(stroke_x[:, 3:5]))


OOL_x, OOL_y = rd.read_dataset("/home/guest/Documents/HAR-Pipeline/DATA/trene", is_stroke=False)
OOL_mean_thigh = np.mean(OOL_x[:, 0:2])
OOL_std_thigh = np.sqrt(np.var(OOL_x[:, 0:2]))

OOL_mean_back = np.mean(OOL_x[:, 3:5])
OOL_std_back = np.sqrt(np.var(OOL_x[:, 3:5]))

IL_x, IL_y = rd.read_dataset("/home/guest/Documents/HAR-Pipeline/DATA/HUNT4-Training-Data-InLab-UpperBackThigh", is_stroke=False)
IL_mean_thigh = np.mean(IL_x[:, 0:2])
IL_std_thigh = np.sqrt(np.var(IL_x[:, 0:2]))

IL_mean_back = np.mean(IL_x[:, 3:5])
IL_std_back = np.sqrt(np.var(IL_x[:, 3:5]))

thigh = False
x = np.arange(-2, 2, 0.01)
if thigh:
    plt.plot(x, mlab.normpdf(x, stroke_mean_thigh, stroke_std_thigh), label= "Stroke")
    plt.plot(x, mlab.normpdf(x, OOL_mean_thigh, OOL_std_thigh), label = "OOL")
    plt.plot(x, mlab.normpdf(x, IL_mean_thigh, IL_std_thigh), label = "IL")
    plt.title("Thigh norm dists")
    plt.legend()
else:
    plt.plot(x, mlab.normpdf(x, stroke_mean_back, stroke_std_back), label="Stroke")
    plt.plot(x, mlab.normpdf(x, OOL_mean_back, OOL_std_back), label="OOL")
    plt.plot(x, mlab.normpdf(x, IL_mean_back, IL_std_back), label="IL")
    plt.title("Back norm dists")
    plt.legend()
plt.show()