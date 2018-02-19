import read_datasets as rd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

subject = "019"
axis = 2
axes = ["x-axis", "y-axis", "z-axis"]

stroke_x, stroke_y = rd.read_dataset("/home/guest/Documents/not_009_011_019", is_stroke=False)
stroke_mean_thigh = np.mean(stroke_x[:, axis])
stroke_std_thigh = np.sqrt(np.var(stroke_x[:, axis]))

stroke_mean_back = np.mean(stroke_x[:, axis+3])
stroke_std_back = np.sqrt(np.var(stroke_x[:, axis+3]))


OOL_x, OOL_y = rd.read_dataset("/home/guest/Documents/009_011_019/"+subject, is_stroke=False)
OOL_mean_thigh = np.mean(OOL_x[:, axis])
OOL_std_thigh = np.sqrt(np.var(OOL_x[:, axis]))

OOL_mean_back = np.mean(OOL_x[:, axis+3])
OOL_std_back = np.sqrt(np.var(OOL_x[:, axis+3]))

#IL_x, IL_y = rd.read_dataset("/home/guest/Documents/HAR-Pipeline/DATA/HUNT4-Training-Data-InLab-UpperBackThigh", is_stroke=False)
#IL_mean_thigh = np.mean(IL_x[:, 0:2])
#IL_std_thigh = np.sqrt(np.var(IL_x[:, 0:2]))

#IL_mean_back = np.mean(IL_x[:, 3:5])
#IL_std_back = np.sqrt(np.var(IL_x[:, 3:5]))


x = np.arange(-2, 2, 0.01)

plt.plot(x, mlab.normpdf(x, stroke_mean_thigh, stroke_std_thigh), label= "regulars")
plt.plot(x, mlab.normpdf(x, OOL_mean_thigh, OOL_std_thigh), label = subject)
plt.axvline(x=stroke_mean_thigh, linestyle='--', color="Red")
#plt.plot(x, mlab.normpdf(x, IL_mean_thigh, IL_std_thigh), label = "IL")
plt.title("THIGH norm dists " + subject + " VS regulars "+ axes[axis])
plt.legend()
plt.show()

plt.plot(x, mlab.normpdf(x, stroke_mean_back, stroke_std_back), label="regulars")
plt.plot(x, mlab.normpdf(x, OOL_mean_back, OOL_std_back), label=subject)
plt.axvline(x=stroke_mean_back, linestyle='--', color="Red")
#plt.plot(x, mlab.normpdf(x, IL_mean_back, IL_std_back), label="IL")
plt.title("BACK norm dists " + subject + " VS regulars " + axes[axis])
plt.legend()
plt.show()