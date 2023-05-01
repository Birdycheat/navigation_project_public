import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import scipy.linalg as scl
import scipy.integrate as sci
import matplotlib
import re

def read_xls(path):
    # convert data stored in path to two dataframes those are respectively the accelerometer
    # and the gyroscope dataframe
    dfs_acc = pd.read_excel(path, sheet_name=0)
    dfs_gyr = pd.read_excel(path, sheet_name=1)
    
    return dfs_acc, dfs_gyr

def df_to_numpy(dfs_acc, dfs_gyr):
    # convert data stored in the dataframe to two numpy arrays containing the following things :
    # - 4 x N array with accelerometer data
    # - 4 x N array with gyroscope data
    
    return dfs_acc.to_numpy().T, dfs_gyr.to_numpy().T

def compute_mean(array):
    # Compute the mean of the 3 x N array along lines
    return np.mean(array, axis = 1)

def compute_var_cov_matrix(array):
    # Compute the 3 x 3 covariance matrix given a 3 x N array
    return np.cov(array)
    
def compute_links(array1, array2):
    n, m = array1.shape[1], array2.shape[1]
    min_len = min(n,m)
    min_max = min(np.max(array1[0,:]), np.max(array2[0,:]))
    
    ret1, ret2 = np.zeros((4, min_len)), np.zeros((4, min_len))
    if min_len == m:
        array1, array2 = array2, array1
        
    argm_prec = 0
    for i in range(ret1.shape[1]):
        if array1[0, i] > min_max + 1e-1:
            break
        ret1[:, i] = array1[:, i]
        argm = np.argmin(np.abs(array2[0,max(0, argm_prec-(int(min_len/10)+1)):(argm_prec+int(min_len/10)+1)] - array1[0, i]))
        ret2[:, i] = array2[:, max(0, argm_prec-(int(min_len/10)+1))+argm]
        argm_prec = max(0, argm_prec-(int(min_len/10)+1))+argm
    if min_len == m:
        array1, array2 = array2, array1
        ret1, ret2 = ret2, ret1
    ret1 = np.delete(ret1, np.s_[i:min_len], 1)
    ret2 = np.delete(ret2, np.s_[i:min_len], 1)
    return ret1, ret2

            
        


def display_df(dfs_acc, dfs_gyr):
    plt.subplots(3, 2)

    plt.subplot(3, 2, 1)
    plt.plot(dfs_acc["Time (s)"], dfs_acc["Acceleration x (m/s^2)"], color="yellow")
    plt.ylabel("Accelerometer x (m/s^2)")
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(dfs_acc["Time (s)"], dfs_acc["Acceleration y (m/s^2)"], color="red")
    plt.ylabel("Accelerometer y (m/s^2)")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(dfs_acc["Time (s)"], dfs_acc["Acceleration z (m/s^2)"], color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Accelerometer z (m/s^2)")
    plt.grid(True)


    plt.subplot(3, 2, 2)
    plt.plot(dfs_gyr["Time (s)"], dfs_gyr["Gyroscope x (rad/s)"], color="yellow", linestyle="--")
    plt.ylabel("Gyroscope x (rad/s)")
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(dfs_gyr["Time (s)"], dfs_gyr["Gyroscope y (rad/s)"], color="red", linestyle="--")
    plt.ylabel("Gyroscope y (rad/s)")
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(dfs_gyr["Time (s)"], dfs_gyr["Gyroscope z (rad/s)"], color="black", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Gyroscope z (rad/s)")
    plt.grid(True)


    plt.show()

def nparray_to_latex(np_array):
    rows, cols = np_array.shape
    
    latex_str = r"\begin{pmatrix}"
    
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            truncated_num = "{:.2e}".format(np_array[i][j])
            row_str += truncated_num
            if j < cols - 1:
                row_str += " & "
        latex_str += row_str
        if i < rows - 1:
            latex_str += r" \\ "
    
    latex_str += r"\end{pmatrix}"
    
    return latex_str

    
def display_np(np_acc, np_gyr): # specialization of the df display function
    plt.subplots(3, 2)

    plt.subplot(3, 2, 1)
    plt.plot(np_acc[0, :], np_acc[1, :], color="yellow")
    plt.ylabel("Accelerometer x (m/s^2)")
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(np_acc[0, :], np_acc[2, :], color="red")
    plt.ylabel("Accelerometer y (m/s^2)")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(np_acc[0, :], np_acc[3, :], color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Accelerometer z (m/s^2)")
    plt.grid(True)


    plt.subplot(3, 2, 2)
    plt.plot(np_gyr[0, :], np_gyr[1, :], color="yellow", linestyle="--")
    plt.ylabel("Gyroscope x (rad/s)")
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(np_gyr[0, :], np_gyr[2, :], color="red", linestyle="--")
    plt.ylabel("Gyroscope y (rad/s)")
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(np_gyr[0, :], np_gyr[3, :], color="black", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Gyroscope z (rad/s)")
    plt.grid(True)


    plt.show()

def display_np_angles(np_gyr):
    plt.subplots(6, 1)
    plt.subplot(6, 1, 1)
    plt.plot(np_gyr[0, :], np_gyr[1, :], color="gold")
    plt.ylabel("$\\theta_1$")
    plt.grid(True)

    plt.subplot(6, 1, 2)
    plt.plot(np_gyr[0, :], np_gyr[2, :], color="red")
    plt.ylabel("$\\theta_2$")
    plt.grid(True)

    plt.subplot(6, 1, 3)
    plt.plot(np_gyr[0, :], np_gyr[3, :], color="black")
    plt.ylabel("$\\theta_3$")
    plt.grid(True)
    
    plt.subplot(6, 1, 4)
    plt.plot(np_gyr[0, :], np_gyr[4, :], color="gold")
    plt.ylabel("$\dot{\\theta_1}$")
    plt.grid(True)

    plt.subplot(6, 1, 5)
    plt.plot(np_gyr[0, :], np_gyr[5, :], color="red")
    plt.ylabel("$\dot{\\theta_2}$")
    plt.grid(True)

    plt.subplot(6, 1, 6)
    plt.plot(np_gyr[0, :], np_gyr[6, :], color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("$\dot{\\theta_3}$")
    plt.grid(True)


    plt.show()

def display_repartition(y, title = " ", one_hot = False):
    classes = np.array(["stairs", "walk", "bike"])
    if one_hot == True:
        freq = y.sum(axis = 0)
        plt.bar(classes, freq)
    else:
        unique, counts = np.unique(y, return_counts=True)
        plt.bar(classes[unique.astype(int)], counts[unique.astype(int)])
    plt.xlabel("Classes")
    plt.ylabel("Fréquence")
    plt.title(title)
    plt.show()

def display_repartition_freq(freq, title = " "):
    classes = np.array(["stairs", "walk", "bike"])
    plt.bar(classes, freq)
    plt.xlabel("Classes")
    plt.ylabel("Fréquence")
    plt.title(title)
    plt.show()


def display_npgyr(np_gyr_filt, labelisation = False, labels = None, one_hot = False):
    L = ["$\\theta_1$", "$\\theta_2$", "$\\theta_3$", "$\\dot{\\theta_1}$", "$\\dot{\\theta_2}$", "$\\dot{\\theta_3}$"]
    colors = ["gold", "red", "black", "purple", "green", "brown"]
    
    for i in range(6):
        plt.subplot(6, 1, i+1)
        plt.plot(np_gyr_filt[0, :], np_gyr_filt[i+1, :], color = colors[i])
        plt.ylabel(L[i])
        plt.grid(True)
    if labelisation == True:
        colors = ["gold", "red", 'black']
        label_list = ["stairs", "walk", "bike"]
        plt.subplot(7, 1, 7)
        if (one_hot == True):
            for p in [np.array([1, 0, 0]),np.array([0, 1, 0]),np.array([0, 0, 1])]:
                index = np.where(p == 1)[0][0]
                indexes = np.where(np.sum(np.logical_not(labels == p), axis = 1) == 0)[0].tolist()
                if len(indexes) == 0:
                    continue
                plt.scatter(np_gyr_filt[0, indexes], np.zeros_like(np_gyr_filt[0, indexes]), label = label_list[index], c = colors[index])
        else:
            for i in range(3):
                plt.scatter(np_gyr_filt[0, labels==i], np.zeros_like(np_gyr_filt[0, labels==i]), label = label_list[i], c = colors[i])
        plt.ylabel("activity")
        plt.legend()
    plt.show()


# State function for the fist EKF
def f_gyro(t, x, Cm_x, Cm_y, Cm_z):
    ##### cst
    L = 0.25*1.8 # 25% of 1.80m
    M = 0.04*80 # 4% of 80kg
    m = 0.14*80 # 14% of 80kg
    r = 0.1
    I_xx = (1/4*m*r**2+1/3*m*L**2)
    I_yy = (1/4*m*r**2+1/3*m*L**2)
    I_zz = 1/2*m*r**2

    # Montreal estimation of gravity vector
    g = 9.806
    #####
    
    x1, x2, x3, x4, x5, x6 = x
    f1 = x4
    f2 = x5
    f3 = x6
    f4 = (2*L**2*np.cos(x2)*np.sin(x2)*x4*x5 + 
          (M+m/2)*L*g*np.cos(x2)*np.sin(x1) + Cm_x(t)) / (I_xx + L**2*np.cos(x2)**2*(M+m/4))
    f5 = ((M+m/2)*L*g*np.cos(x1)*np.sin(x2) + Cm_y(t)) / (I_yy + L**2*(M+m/4))
    f6 = Cm_z(t) / I_zz
    return [f1, f2, f3, f4, f5, f6]



def numpy_ftot(t, x, Cmx, Cmy, Cmz, L=0.25*1.8, M=0.04*80, m=0.14*80, g=9.806, r=0.1, LS = 0.20):
    
    Ixx = (1/4*m*r**2+1/3*m*L**2)
    Iyy = (1/4*m*r**2+1/3*m*L**2)
    Izz = 1/2*m*r**2
    
    
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
    def numpy_fdottheta(x1, x2, x3, x4, x5, x6):
        f4 = (2 * L ** 2 * np.cos(x2) * np.sin(x2) * x4 * x5 +
              (M + m / 2) * L * g * np.cos(x2) * np.sin(x1) + Cmx(t)) / (Ixx + L ** 2 * np.cos(x2) ** 2 * (M + m / 4))
        f5 = ((M + m / 2) * L * g * np.cos(x1) * np.sin(x2) + Cmy(t)) / (Iyy + L ** 2 * (M + m / 4))
        f6 = Cmz(t) / Izz
        return np.array([f4, f5, f6])

    def numpy_cross_product(theta1, theta2, theta3):
        return np.array([[0, -theta3, theta2],
                         [theta3, 0, -theta1],
                         [-theta2, theta1, 0]])

    def numpy_Rx(x1):
        return np.array([[1, 0, 0],
                         [0, np.cos(x1), -np.sin(x1)],
                         [0, np.sin(x1), np.cos(x1)]])

    def numpy_Ry(x2):
        return np.array([[np.cos(x2), 0, np.sin(x2)],
                         [0, 1, 0],
                         [-np.sin(x2), 0, np.cos(x2)]])

    def numpy_Rz(x3):
        return np.array([[np.cos(x3), -np.sin(x3), 0],
                         [np.sin(x3), np.cos(x3), 0],
                         [0, 0, 1]])

    def numpy_R0rod(x1, x2, x3):
        return np.matmul(numpy_Rx(x1), np.matmul(numpy_Ry(x2), numpy_Rz(x3)))

    def numpy_Rsr():
        return np.array([[-1, 0, 0],
                         [0, 0, -1],
                         [0, -1, 0]])

    ddtheta = numpy_fdottheta(x1, x2, x3, x4, x5, x6)
    dddtheta = numpy_fdottheta(x4, x5, x6, ddtheta[0], ddtheta[1], ddtheta[2])

    cross_dtheta = numpy_cross_product(x4, x5, x6)
    cross_ddtheta = numpy_cross_product(ddtheta[0], ddtheta[1], ddtheta[2])
    cross_dddtheta = numpy_cross_product(dddtheta[0], dddtheta[1], dddtheta[2])

    k456 = np.matmul(numpy_Rsr(),
                     np.matmul(np.transpose(numpy_R0rod(x1, x2, x3)), cross_dtheta) +
                     np.matmul(cross_dtheta, np.matmul(cross_dtheta, cross_dtheta)) +
                     2 * np.matmul(cross_dtheta, cross_ddtheta) + np.matmul(cross_ddtheta, cross_dtheta) + cross_dddtheta) @ np.array([[0], [r], [-LS]])
    k = np.array([k456[0, 0], k456[1, 0], k456[2, 0]])

    f1, f2, f3 = numpy_fdottheta(x1, x2, x3, x4, x5, x6)

    fk = np.array([x4, x5, x6, f1, f2, f3, k[0], k[1], k[2]])

    return fk


def jacobian_ftot(x, u, L=0.25*1.8, M=0.04*80, m=0.14*80, g=9.806, r=0.1, LS = 0.20):
    Ixx = (1/4*m*r**2+1/3*m*L**2)
    Iyy = (1/4*m*r**2+1/3*m*L**2)
    Izz = 1/2*m*r**2
    
    
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
    
    Cmx, Cmy, Cmz = u
    return np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [L*g*(M + m/2)*np.cos(x1)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2), 2*L**2*(M + m/4)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)**2 + (-2*L**2*x4*x5*np.sin(x2)**2 + 2*L**2*x4*x5*np.cos(x2)**2 - L*g*(M + m/2)*np.sin(x1)*np.sin(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2), 0, 2*L**2*x5*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2), 2*L**2*x4*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2), 0, 0, 0, 0], [-L*g*(M + m/2)*np.sin(x1)*np.sin(x2)/(Iyy + L**2*(M + m/4)), L*g*(M + m/2)*np.cos(x1)*np.cos(x2)/(Iyy + L**2*(M + m/4)), 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [-LS*(-2*L*g*x6*(M + m/2)*np.cos(x1)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) + x4*np.sin(x2)*np.cos(x1)*np.cos(x3)) + r*(L*g*x4*(M + m/2)*np.sin(x1)*np.sin(x2)/(Iyy + L**2*(M + m/4)) - 2*L*g*x5*(M + m/2)*np.cos(x1)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - x4*np.sin(x1)*np.sin(x2)*np.cos(x3)), -LS*(-4*L**2*x6*(M + m/4)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)**2 + x4*np.sin(x1)*np.cos(x2)*np.cos(x3) + x5*np.sin(x2)*np.cos(x3) - 2*x6*(-2*L**2*x4*x5*np.sin(x2)**2 + 2*L**2*x4*x5*np.cos(x2)**2 - L*g*(M + m/2)*np.sin(x1)*np.sin(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)) + r*(-4*L**2*x5*(M + m/4)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)**2 - L*g*x4*(M + m/2)*np.cos(x1)*np.cos(x2)/(Iyy + L**2*(M + m/4)) + x4*np.cos(x1)*np.cos(x2)*np.cos(x3) - 2*x5*(-2*L**2*x4*x5*np.sin(x2)**2 + 2*L**2*x4*x5*np.cos(x2)**2 - L*g*(M + m/2)*np.sin(x1)*np.sin(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - x6*np.sin(x2)*np.cos(x3)), -LS*(-x4*np.sin(x1)*np.sin(x2)*np.sin(x3) + x5*np.sin(x3)*np.cos(x2)) + r*(-x4*np.sin(x2)*np.sin(x3)*np.cos(x1) - x6*np.sin(x3)*np.cos(x2)), -LS*(-Cmz/Izz - 4*L**2*x5*x6*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) + L*g*(M + m/2)*np.sin(x4)*np.sin(x5)/(Iyy + L**2*(M + m/4)) + 2*x4*x5 + np.sin(x1)*np.sin(x2)*np.cos(x3)) + r*(-4*L**2*x5**2*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - 2*x4*x6 - (Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))/(Iyy + L**2*(M + m/4)) + np.sin(x2)*np.cos(x1)*np.cos(x3)), -LS*(-4*L**2*x4*x6*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - L*g*(M + m/2)*np.cos(x4)*np.cos(x5)/(Iyy + L**2*(M + m/4)) + x4**2 + 3*x5**2 + x6**2 - np.cos(x2)*np.cos(x3)) + r*(-4*L**2*x4*x5*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - 2*x5*x6 - 2*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)), -LS*(2*x5*x6 - 2*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)) + r*(-x4**2 - x5**2 - 3*x6**2 + np.cos(x2)*np.cos(x3)), 0, 0, 0], [-LS*(3*L*g*x4*(M + m/2)*np.cos(x1)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - 3*L*g*x5*(M + m/2)*np.sin(x1)*np.sin(x2)/(Iyy + L**2*(M + m/4)) - x4*np.cos(x1)*np.cos(x2)) + r*(L*g*x6*(M + m/2)*np.sin(x1)*np.sin(x2)/(Iyy + L**2*(M + m/4)) + x4*np.sin(x1)*np.cos(x2) - (2*L**3*g*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(M + m/2)*np.sin(x5)*np.cos(x1)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) - 2*L**3*g*(M + m/2)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x1)*np.sin(x2)*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))))/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)), -LS*(6*L**2*x4*(M + m/4)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)**2 + 3*L*g*x5*(M + m/2)*np.cos(x1)*np.cos(x2)/(Iyy + L**2*(M + m/4)) + x4*np.sin(x1)*np.sin(x2) + 3*x4*(-2*L**2*x4*x5*np.sin(x2)**2 + 2*L**2*x4*x5*np.cos(x2)**2 - L*g*(M + m/2)*np.sin(x1)*np.sin(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - x5*np.cos(x2)) + r*(-L*g*x6*(M + m/2)*np.cos(x1)*np.cos(x2)/(Iyy + L**2*(M + m/4)) + x4*np.sin(x2)*np.cos(x1) + x6*np.cos(x2) - (4*L**4*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(M + m/4)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x2)*np.sin(x5)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)**2*(Iyy + L**2*(M + m/4))) + 2*L**3*g*(M + m/2)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x5)*np.cos(x1)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + 2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(-2*L**2*x4*x5*np.sin(x2)**2 + 2*L**2*x4*x5*np.cos(x2)**2 - L*g*(M + m/2)*np.sin(x1)*np.sin(x2))*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))))/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)), 0, -LS*(6*L**2*x4*x5*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - np.sin(x1)*np.cos(x2) + 3*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)) + r*(3*x4**2 + x5**2 + x6**2 - np.cos(x1)*np.cos(x2) - (4*L**4*x5*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*np.sin(x2)*np.sin(x5)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + L*g*(M + m/2)*np.cos(x4)*np.cos(x5))/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)), -LS*(6*L**2*x4**2*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) + 3*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))/(Iyy + L**2*(M + m/4)) - np.sin(x2)) + r*(-2*Cmz/Izz - 2*L**2*(M + m/4)*(Cmx + 2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + L*g*(M + m/2)*np.sin(x4)*np.cos(x5))*np.sin(x5)*np.cos(x5)/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)**2 + 2*x4*x5 - (4*L**4*x4*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*np.sin(x2)*np.sin(x5)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) - 2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x5)**2/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + 2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.cos(x5)**2/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) - L*g*(M + m/2)*np.sin(x4)*np.sin(x5))/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)), r*(2*x4*x6 - (Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))/(Iyy + L**2*(M + m/4)) + np.sin(x2)), 0, 0, 0], [-LS*(2*L*g*x6*(M + m/2)*np.sin(x1)*np.sin(x2)/(Iyy + L**2*(M + m/4)) - x4*np.sin(x2)*np.sin(x3)*np.cos(x1) + (2*L**3*g*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(M + m/2)*np.sin(x5)*np.cos(x1)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) - 2*L**3*g*(M + m/2)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x1)*np.sin(x2)*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))))/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)) + r*(3*L*g*x4*(M + m/2)*np.cos(x1)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) + x4*np.sin(x1)*np.sin(x2)*np.sin(x3)), -LS*(-2*L*g*x6*(M + m/2)*np.cos(x1)*np.cos(x2)/(Iyy + L**2*(M + m/4)) - x4*np.sin(x1)*np.sin(x3)*np.cos(x2) - x5*np.sin(x2)*np.sin(x3) + (4*L**4*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(M + m/4)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x2)*np.sin(x5)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)**2*(Iyy + L**2*(M + m/4))) + 2*L**3*g*(M + m/2)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x5)*np.cos(x1)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + 2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(-2*L**2*x4*x5*np.sin(x2)**2 + 2*L**2*x4*x5*np.cos(x2)**2 - L*g*(M + m/2)*np.sin(x1)*np.sin(x2))*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))))/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)) + r*(6*L**2*x4*(M + m/4)*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)**2 - x4*np.sin(x3)*np.cos(x1)*np.cos(x2) + 3*x4*(-2*L**2*x4*x5*np.sin(x2)**2 + 2*L**2*x4*x5*np.cos(x2)**2 - L*g*(M + m/2)*np.sin(x1)*np.sin(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) + x6*np.sin(x2)*np.sin(x3)), -LS*(-x4*np.sin(x1)*np.sin(x2)*np.cos(x3) + x5*np.cos(x2)*np.cos(x3)) + r*(-x4*np.sin(x2)*np.cos(x1)*np.cos(x3) - x6*np.cos(x2)*np.cos(x3)), -LS*(-3*x4**2 - x5**2 - x6**2 - np.sin(x1)*np.sin(x2)*np.sin(x3) + (4*L**4*x5*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*np.sin(x2)*np.sin(x5)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + L*g*(M + m/2)*np.cos(x4)*np.cos(x5))/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)) + r*(6*L**2*x4*x5*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - np.sin(x2)*np.sin(x3)*np.cos(x1) + 3*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)), 6*L**2*r*x4**2*np.sin(x2)*np.cos(x2)/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - LS*(-Cmz/Izz + 2*L**2*(M + m/4)*(Cmx + 2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + L*g*(M + m/2)*np.sin(x4)*np.cos(x5))*np.sin(x5)*np.cos(x5)/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)**2 - 2*x4*x5 + np.sin(x3)*np.cos(x2) + (4*L**4*x4*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*np.sin(x2)*np.sin(x5)*np.cos(x2)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) - 2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x5)**2/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + 2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.cos(x5)**2/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) - L*g*(M + m/2)*np.sin(x4)*np.sin(x5))/(Ixx + L**2*(M + m/4)*np.cos(x5)**2)), -LS*(-2*x4*x6 - 2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))/(Iyy + L**2*(M + m/4))) + r*(3*Cmz/Izz - np.sin(x3)*np.cos(x2)), 0, 0, 0]])


def jacobian_ftot_cmd(x, u, L=0.25*1.8, M=0.04*80, m=0.14*80, g=9.806, r=0.1, LS = 0.20):
    Ixx = (1/4*m*r**2+1/3*m*L**2)
    Iyy = (1/4*m*r**2+1/3*m*L**2)
    Izz = 1/2*m*r**2
    
    
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
    
    Cmx, Cmy, Cmz = u
    
    return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1/(Ixx + L**2*(M + m/4)*np.cos(x2)**2), 0, 0], [0, 1/(Iyy + L**2*(M + m/4)), 0], [0, 0, 1/Izz], [2*LS*x6/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - 2*r*x5/(Ixx + L**2*(M + m/4)*np.cos(x2)**2), LS/(Iyy + L**2*(M + m/4)) - r*x4/(Iyy + L**2*(M + m/4)), LS*x4/Izz + r/Izz], [-3*LS*x4/(Ixx + L**2*(M + m/4)*np.cos(x2)**2) - r*(2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + 1)/(Ixx + L**2*(M + m/4)*np.cos(x5)**2), -3*LS*x5/(Iyy + L**2*(M + m/4)) + r*(-2*L**2*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Ixx + L**2*(M + m/4)*np.cos(x5)**2)*(Iyy + L**2*(M + m/4))) - x6/(Iyy + L**2*(M + m/4))), -2*r*x5/Izz], [-LS*(2*L**2*(Cmy + L*g*(M + m/2)*np.sin(x2)*np.cos(x1))*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Iyy + L**2*(M + m/4))) + 1)/(Ixx + L**2*(M + m/4)*np.cos(x5)**2) + 3*r*x4/(Ixx + L**2*(M + m/4)*np.cos(x2)**2), -LS*(2*L**2*(Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2))*np.sin(x5)*np.cos(x5)/((Ixx + L**2*(M + m/4)*np.cos(x2)**2)*(Ixx + L**2*(M + m/4)*np.cos(x5)**2)*(Iyy + L**2*(M + m/4))) - 2*x6/(Iyy + L**2*(M + m/4))), LS*x5/Izz + 3*r*x6/Izz]])

def jacobian_gyro(x, u, L, M, m, Ixx, Iyy, Izz, g):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    
    Cmx = u[0]
    Cmy = u[1]
    Cmz = u[2]
    
    J = np.zeros((6, 6))
    J[0, 3] = 1
    J[1, 4] = 1
    J[2, 5] = 1
    
    a = L * g * (M + m/2) * np.cos(x1) * np.cos(x2) / (Ixx + L**2 * (M + m/4) * np.cos(x2)**2)
    b = 2 * L**2 * (M + m/4) * (Cmx + 2*L**2*x4*x5*np.sin(x2)*np.cos(x2) + L*g*(M + m/2)*np.sin(x1)*np.cos(x2)) * np.sin(x2)*np.cos(x2) / (Ixx + L**2 * (M + m/4) * np.cos(x2)**2)**2 + (-2*L**2*x4*x5*np.sin(x2)**2 + 2*L**2*x4*x5*np.cos(x2)**2 - L*g*(M + m/2)*np.sin(x1)*np.sin(x2))/(Ixx + L**2*(M + m/4)*np.cos(x2)**2)
    c = (2*L**2*x5*np.sin(x2)*np.cos(x2)) / (Ixx + L**2 * (M + m/4) * np.cos(x2)**2)

    J[3, 0] = a
    J[3, 1] = b
    J[3, 3] = c
    J[3, 4] = 2 * L**2 * x4 * np.sin(x2) * np.cos(x2) / (Ixx + L**2 * (M + m/4) * np.cos(x2)**2)
    J[4, 0] = -L * g * (M + m/2) * np.sin(x1) * np.sin(x2) / (Iyy + L**2 * (M + m/4))
    J[4, 1] = L * g * (M + m/2) * np.cos(x1) * np.cos(x2) / (Iyy + L**2 * (M + m/4))
    
    return J

def jacobian_gyro_cmd(x, u, L, M, m, Ixx, Iyy, Izz, g):
    Cmx, Cmy, Cmz = u
    jac = np.zeros((6,3))
    jac[3,0] = 1/(Ixx + L**2*(M + m/4)*np.cos(x[1])**2)
    jac[4,1] = 1/(Iyy + L**2*(M + m/4))
    jac[5,2] = 1/Izz
    return jac


def discretize_state(A, B, L, tk, tkp1, Q = None):
    Ak = scl.expm(A*(tkp1 - tk))
    integrand = lambda tau: (scl.expm(A*(tkp1 - tau))@B).reshape((A.shape[0]*B.shape[1],))
    Bk = (sci.quad_vec(integrand, tk, tkp1)[0]).reshape((A.shape[0], B.shape[1]))
    if (L != None and Q != None):
        integrandQ = lambda s: (scl.expm(A*s)@L@Q@L.T@scl.expm(A*s).T).reshape((A.shape[0]**2))
        Qk = (sci.quad_vec(integrandQ, 0, tkp1-tk)[0]).reshape(A.shape)
    else:
        Qk = None
    return Ak, Bk, Qk


def discretize_state_first_order(A, B, L, tk, tkp1, Q = None): # it is second order...
    hk = tkp1 - tk
    Ak = np.eye(A.shape[0]) + A*hk + A@A*hk**2/2
    Bk = ((np.eye(A.shape[0]) + A*hk/2)*hk)@B
    Qk = L@Q@L.T*hk + 0.5*(L@Q@L.T@A.T + A@L@Q@L.T)*hk**2
    return Ak, Bk, Qk


def latex_matrix_to_numpy_array(latex_matrix_str):
    rows = [row.strip() for row in latex_matrix_str.split("\\\\")]
    return np.array([list(map(float, row.split("&"))) for row in rows])

def read_matrices_from_file(file_path):
    with open(file_path, 'r') as file:
        latex_string = file.read()

    # Find the matrices in the latex string
    matrices = re.findall(r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}", latex_string, re.DOTALL)

    # Convert each matrix found to a np.array
    numpy_arrays = [latex_matrix_to_numpy_array(matrix) for matrix in matrices]

    return numpy_arrays
    

def drop_first_and_last(dfs_acc, dfs_gyr, tbeg, tend):
    dfs_acc = dfs_acc[dfs_acc['Time (s)'] >= tbeg]
    dfs_gyr = dfs_gyr[dfs_gyr['Time (s)'] >= tbeg]

    dfs_acc = dfs_acc[dfs_acc['Time (s)'] <= tend]
    dfs_gyr = dfs_gyr[dfs_gyr['Time (s)'] <= tend]

    # Reset index after filtering
    dfs_acc.reset_index(drop=True, inplace=True)
    dfs_gyr.reset_index(drop=True, inplace=True)
    return dfs_acc, dfs_gyr


def drop_first_and_last_np(np_gyr, tbeg, tend):
    np_gyr = np_gyr[:, np_gyr[0, :] >= tbeg]
    np_gyr = np_gyr[:, np_gyr[0, :] <= tend]

    return np_gyr

def split_dataframes(dfs_acc, dfs_gyr, tsplit):
    # Find the nearest time index in dfs_acc
    acc_split_index = dfs_acc.iloc[(dfs_acc['Time (s)'] - tsplit).abs().argsort()[:1]].index[0]

    # Find the nearest time index in dfs_gyr
    gyr_split_index = dfs_gyr.iloc[(dfs_gyr['Time (s)'] - tsplit).abs().argsort()[:1]].index[0]

    # Split dfs_acc
    dfs_acc_1 = dfs_acc.iloc[:acc_split_index]
    dfs_acc_2 = dfs_acc.iloc[acc_split_index:]

    # Split dfs_gyr
    dfs_gyr_1 = dfs_gyr.iloc[:gyr_split_index]
    dfs_gyr_2 = dfs_gyr.iloc[gyr_split_index:]
    
    dfs_acc_1.reset_index(drop=True, inplace=True)
    dfs_gyr_1.reset_index(drop=True, inplace=True)
    dfs_acc_2.reset_index(drop=True, inplace=True)
    dfs_gyr_2.reset_index(drop=True, inplace=True)

    return (dfs_acc_1, dfs_gyr_1), (dfs_acc_2, dfs_gyr_2)

def load_to_array(path):
    out = np.load(path)
    array = np.block([[out.f.t], [out.f.x]])
    return array

def split_array(gyr_array, tsplit):
    # Find the nearest time index in gyr_array
    gyr_split_index = np.abs(gyr_array[0, :] - tsplit).argmin()

    # Split gyr_array
    gyr_array_1 = gyr_array[:, :gyr_split_index]
    gyr_array_2 = gyr_array[:, gyr_split_index:]

    return gyr_array_1, gyr_array_2

def np_save(np_gyr, name):
    np.savez(name, t = np_gyr[0, :], x = np_gyr[1:, :])