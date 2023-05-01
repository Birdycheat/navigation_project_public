import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from utils import read_xls, df_to_numpy, read_matrices_from_file, jacobian_ftot, jacobian_ftot_cmd, discretize_state_first_order, compute_links
from Kalman_filter import EKF_gyro
    


def EKF_tot(ts, output, R, P0, Q, L_mat, Cm_x, Cm_y, Cm_z, L = 0.25*1.8, M = 0.04*80, m = 0.14*80, r = 0.1, g = 9.806, LS = 0.20, display_traj = True, display_cov = True, save = False, save_path = None):
    
    dH_tot = np.eye(9)
    # init
    x = output[:, 0]
    x_pred = np.zeros((9, output.shape[1]))
    x_pred[:, 0] = x.reshape((9,))
    x_filt = np.zeros_like(x_pred)
    x_filt[:, 0] = x.reshape((9,))
    R_inv = np.linalg.inv(R)
    
    Vkp1_k = P0
    Vkp1 = P0
    Vks = np.zeros((9, 9, ts.size))
    Vks[:, :, 0] = P0

    # EKF loop

    for n in range(output.shape[1]-1):
        u = np.array([Cm_x(ts[n]), Cm_y(ts[n]), Cm_z(ts[n])])
        # Prediction
        A = jacobian_ftot(x_filt[:, n], u)
        B = jacobian_ftot_cmd(x_filt[:, n], u)
        Ak, Bk, Qk = discretize_state_first_order(A, B, L_mat, ts[n], ts[n+1], Q)
        x_pred[:, n+1] = Ak @ x_filt[:, n] + Bk @ u
        
        Vkp1_k = Ak @ Vkp1 @ Ak.T + L_mat @ Qk @ L_mat.T
        Vkp1 = Vkp1_k - Vkp1_k @ dH_tot.T @ np.linalg.inv(dH_tot @ Vkp1_k @ dH_tot.T + R) @ dH_tot @ Vkp1_k
        Vks[:, :, n+1] = Vkp1
        # Correction
        y = output[:, n+1]
        K = Vkp1 @ dH_tot.T @ R_inv
        x_filt[:, n+1] = x_pred[:, n+1] + K @ (y - dH_tot@x_pred[:, n+1])


    if display_traj == True:
        output = dH_tot.T@output 
        plt.subplots(3, 3)
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            plt.plot(ts, x_filt[i-1, :], label = "filtred $x_{" + str(i) + "}$")
            plt.plot(ts, x_pred[i-1, :], label = "predicted $x_{" + str(i) + "}$")
            plt.plot(ts, output[i-1, :], label = "true $x_{"+ str(i) + "}$")
            plt.legend()
        plt.show()

    if display_cov == True:
        plt.subplots(9, 9, sharex=True, sharey=True)


        for i in range(1, 10):
            for j in range(1, 10):
                plt.subplot(9, 9, j + 9*(i-1))
                plt.plot(ts, Vks[i-1, j-1], label = "$P_{" + str(i) + str(j) + "}$")
                
                plt.legend()
        plt.show()
        
    if save == True:
        if type(save_path) != None:
            np.savez(save_path, t = ts, x = x_filt)
        else:
            exit("Give a file name to save your filtered datas")
            
    return ts, x_filt




# Initial covariance matrix for the first EFK
P0 = np.diag([0.1, 0.1, 0.1, 1, 1, 1]) 
# Initial covariance matrix for the second EFK
PP0 = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1])

# Model noise for the first EKF
Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])**2
# Model noise for the second EKF
QQ = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])**2


file_path = "means_covs.md"
matrices = read_matrices_from_file(file_path)
name_list = ["mu_AGacc", "Gamma_AGacc", "mu_AGgyr", "Gamma_AGgyr","mu_GBacc", "Gamma_GBacc" , "mu_GBgyr", "Gamma_GBgyr","mu_MDacc", "Gamma_MDacc" , "mu_MDgyr", "Gamma_MDgyr"]
dict = {}


for i, matrix in enumerate(matrices):
    dict[name_list[i]] = matrix

# Inputs torques 
Cm_x = lambda t: 0.1*np.cos(6*t)*np.exp(-10*t)
Cm_y = lambda t: 0.1*np.cos(6*0.8*t)*np.exp(-10*t)
Cm_z = lambda t: 0

L_mat = np.diag([1, 1, 1, 1, 1, 1]) # Process noise matrix for the first EKF
LL_mat = np.diag([10, 10, 100, 1, 1, 1, 1, 1, 1]) # ... for the second 

names = ['AG', 'MD', 'GB']
extension = ["xls", "xlsx"]

for name in names:
    directory = "Data/" + name
    for file in listdir(directory):
        if file.split(".")[-1] in extension:
            # Set the measurements covariance matrices
            R_gyr = dict["Gamma_" + name + "gyr"]
            R_acc = dict["Gamma_" + name + "acc"]
            
            # Set the measurements bias vectors
            mu_gyr = dict["mu_" + name + "gyr"]
            mu_acc = dict["mu_" + name + "acc"]
            
            path = directory + '/' + file
            print(path)
            # Loading measurements
            dfs_acc, dfs_gyr = read_xls(path)

            np_acc, np_gyr = df_to_numpy(dfs_acc, dfs_gyr)
            np_acc, np_gyr = compute_links(np_acc, np_gyr)
            
            # Process centering (under the assumption of independent noise over time)
            output_gyr, output_acc = np_gyr[1:, :] - mu_gyr, np_acc[1:, :] - mu_acc
            save_path = path.split(".")[0] + ".npz"
            
            # first filter that outputs angles and velocities
            _, theta_filt, Vk_theta = EKF_gyro(np_gyr[0, :], output_gyr, R_gyr, P0, Q, L_mat, Cm_x, Cm_y, Cm_z, save=False, save_path=save_path, display_cov=True, display_traj=True)
            
            R = np.block([[Vk_theta, np.zeros((6, 3))],
                          [np.zeros((3, 6)), R_acc]])
            ts = (np_gyr[0, :] + np_acc[0, :])/2
            output = np.block([[theta_filt], [output_acc]])
            
            ts, x_filt = EKF_tot(ts, output, R, PP0, QQ, LL_mat, Cm_x, Cm_y, Cm_z, save=True, save_path=save_path, display_cov=True, display_traj=True)
        