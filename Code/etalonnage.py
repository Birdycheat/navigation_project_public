import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import display_np, read_xls, df_to_numpy, compute_mean, compute_var_cov_matrix, nparray_to_latex
import os

# Program that computes covariances and means for signals for 15s < t < 45s

g = 9.806

t_beg = 15
t_end = 45

directory = "Data/Etalonnage"
with open('means_covs.md', "w") as fichier_md:
    for folder in os.listdir(directory):
        cov_glob_acc, cov_glob_gyr = np.zeros((3, 3)), np.zeros((3, 3))
        mean_glob_acc, mean_glob_gyr = np.zeros((3)), np.zeros((3))
        for filename in os.listdir(directory+"/"+folder):
            f = directory+"/"+folder+"/"+filename
            if (os.path.isfile(f) and "zero" in filename.lower()):
                np_acc, dfs_gyr = read_xls(f)
                np_acc, np_gyr_filt = df_to_numpy(np_acc, dfs_gyr)
                
                np_acc, np_gyr_filt = np_acc[:, (np_acc[0, :] >= t_beg)*(np_acc[0, :] <= t_end)], np_gyr_filt[:, (np_gyr_filt[0, :] >= t_beg)*(np_gyr_filt[0, :] <= t_end)]
                
                
                print(f)
                mean_acc = compute_mean(np_acc[1:,:])
                mean_glob_acc += mean_acc
                cov_acc = compute_var_cov_matrix(np_acc[1:,:])
                cov_glob_acc += cov_acc

                mean_gyr = compute_mean(np_gyr_filt[1:,:])
                mean_glob_gyr += mean_gyr
                
                cov_gyr = compute_var_cov_matrix(np_gyr_filt[1:,:])
                cov_glob_gyr += cov_gyr

        bias_acc = (mean_glob_acc - g)/3
        bias_gyr = mean_glob_gyr/3
        cov_glob_acc/=3
        cov_glob_gyr/=3
        
        fichier_md.write("$\\mu^{"+"\\text{acc}}_{"+"\\text{"+folder+"}}"+"={}".format(nparray_to_latex(bias_acc.reshape((-1, 1)))))
        fichier_md.write(", \\Gamma^{"+"\\text{acc}}_{"+"\\text{"+folder+"}}="+"{}".format(nparray_to_latex(cov_glob_acc)))
        fichier_md.write("$\n\n$\\mu^{"+"\\text{gyr}}_{"+"\\text{"+folder+"}}"+"={}".format(nparray_to_latex(bias_gyr.reshape((-1, 1)))))
        fichier_md.write(", \\Gamma^{"+"\\text{gyr}}_{"+"\\text{"+folder+"}}"+"={}".format(nparray_to_latex(cov_glob_gyr))+"$\n\n\n\n")