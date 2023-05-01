import numpy as np
import matplotlib.pyplot as plt
from utils import display_np, read_xls, df_to_numpy, compute_mean, compute_var_cov_matrix, compute_links
import os


directory = "Data"
for folder in os.listdir(directory):
    for filename in os.listdir(directory+"/"+folder):
        f = directory+"/"+folder+"/"+filename
        if os.path.isfile(f) and f.split(".")[-1] in ["xls", "xls"]:
            np_acc, dfs_gyr = read_xls(f)
            np_acc, np_gyr = df_to_numpy(np_acc, dfs_gyr)
            
            
            print(f)
            np_acc, np_gyr = compute_links(np_acc, np_gyr)
            print("Delta max gyro/accel: {}".format(np.max(np.abs(np_acc[0, :] - np_gyr[0, :]))))
            display_np(np_acc, np_gyr)