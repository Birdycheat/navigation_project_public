import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from utils import read_xls, df_to_numpy, display_df, read_matrices_from_file, f_gyro, jacobian_gyro, jacobian_gyro_cmd, discretize_state_first_order



def EKF_gyro(ts, output, R, P0, Q, L_mat, Cm_x, Cm_y, Cm_z, L = 0.25*1.8, M = 0.04*80, m = 0.14*80, r = 0.1, g = 9.806, display_traj = True, display_cov = True, save = False, save_path = None):
    
    # Constants definitions
    I_xx = (1/4*m*r**2+1/3*m*L**2)
    I_yy = (1/4*m*r**2+1/3*m*L**2)
    I_zz = 1/2*m*r**2
    
    # Jacobian of the output
    dH = np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [-1, 0, 0],
                [0, 0, -1],
                [0, -1, 0]]).T
    # init
    x = np.concatenate((np.zeros((3,)), output[:, 0]))
    x_pred = np.zeros((6, output.shape[1]))
    x_pred[:, 0] = x.reshape((6,))
    x_filt = np.zeros_like(x_pred)
    x_filt[:, 0] = x.reshape((6,))
    R_inv = np.linalg.inv(R)
    
    Vkp1_k = P0
    Vkp1 = P0
    Vks = np.zeros((6, 6, ts.size))
    Vks[:, :, 0] = P0

    # EKF loop

    for n in range(output.shape[1]-1):
        u = np.array([Cm_x(ts[n]), Cm_y(ts[n]), Cm_z(ts[n])])
        # Prediction
        A = jacobian_gyro(x_filt[:, n], u, L, M, m, I_xx, I_yy, I_zz, g)
        B = jacobian_gyro_cmd(x_filt[:, n], u, L, M, m, I_xx, I_yy, I_zz, g)
        Ak, Bk, Qk = discretize_state_first_order(A, B, L_mat, ts[n], ts[n+1], Q)
        x_pred[:, n+1] = Ak @ x_filt[:, n] + Bk @ u
        
        Vkp1_k = Ak @ Vkp1 @ Ak.T + L_mat @ Qk @ L_mat.T
        Vkp1 = Vkp1_k - Vkp1_k @ dH.T @ np.linalg.inv(dH @ Vkp1_k @ dH.T + R) @ dH @ Vkp1_k
        Vks[:, :, n+1] = Vkp1
        # Correction
        y = output[:, n+1]
        K = Vkp1 @ dH.T @ R_inv
        x_filt[:, n+1] = x_pred[:, n+1] + K @ (y - dH@x_pred[:, n+1])


    if display_traj == True:
        output = dH.T@output 
        plt.subplots(6, 1)
        for i in range(1, 7):
            plt.subplot(6, 1, i)
            plt.plot(ts, x_filt[i-1, :], label = "$x_{}$ filtred".format(i))
            plt.plot(ts, x_pred[i-1, :], label = "$x_{}$ predicted".format(i))
            
            if i >= 4:
                plt.plot(ts, output[i-1, :], label = "true $x_{}$".format(i-1))
            plt.legend()
        plt.show()

    if display_cov == True:
        plt.subplots(6, 6, sharex=True, sharey=True)


        for i in range(1, 7):
            for j in range(1, 7):
                plt.subplot(6, 6, j + 6*(i-1))
                plt.plot(ts, Vks[i-1, j-1], label = "$P_{" + str(i) + str(j) + "}$")
                
                plt.legend()
        plt.show()
        
    if save == True:
        if type(save_path) != None:
            np.savez(save_path, t = ts, x = x_filt)
        else:
            exit("Give a file name to save your filtered datas")
            
    return ts, x_filt, Vks[:, :, -1]
    
    



def main():
    # initial conditions
    P0 = np.diag([0.1, 0.1, 0.1, 1, 1, 1])  # Initial covariance matrix

    # Model noise
    Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])**2

    file_path = "means_covs.md"
    matrices = read_matrices_from_file(file_path)
    name_list = ["mu_AGgyr", "Gamma_AGgyr", "mu_AGacc", "Gamma_AGacc","mu_GBgyr", "Gamma_GBgyr" , "mu_GBacc", "Gamma_GBacc","mu_MDgyr", "Gamma_MDgyr" , "mu_MDacc", "Gamma_MDacc"]
    dict = {}


    for i, matrix in enumerate(matrices):
        dict[name_list[i]] = matrix

    # Torques inputs
    Cm_x = lambda t: 0.01*np.cos(6*t)*np.exp(-10*t)
    Cm_y = lambda t: 0.01*np.cos(6*0.8*t)*np.exp(-10*t)
    Cm_z = lambda t: 0.01*np.cos(6*0.8*t)*np.exp(-10*t)*0

    # Definition of the model noise matrix

    L_mat = np.diag([1, 1, 1, 1, 1, 1])

    names = ['AG', 'MD', 'GB']
    extension = ["xls", "xlsx"]

    for name in names:
        directory = "Data/" + name
        for file in listdir(directory):
            if file.split(".")[-1] in extension:
                R = dict["Gamma_" + name + "gyr"]
                path = directory + '/' + file
                print(path)
                # Loading of measurements
                _, dfs_gyr = read_xls(path)

                _, np_gyr_filt = df_to_numpy(dfs_gyr, dfs_gyr)

                output = np_gyr_filt[1:, :]
                ts = np_gyr_filt[0, :]
                save_path = path.split(".")[0] + ".npz"
                EKF_gyro(ts, output, R, P0, Q, L_mat, Cm_x, Cm_y, Cm_z, save=False, save_path=save_path, display_cov=True, display_traj=True)
        
if __name__ == "__main__":
    main()