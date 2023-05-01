**FIRST AND FOREMOST, TO EXECUTE THE DIFFERENT CODES YOU MUST BE IN THE Navigation_project FOLDER.**

This file aims to explain the functionality of each of the code files and data management.
- **Matlab folder**: This folder contains the Matlab code for performing classification using LSTM recurrent neural networks. It is with these algorithms that we obtain the best accuracy, and these are the ones presented in the report. **The models mentioned in the report are saved with their respective window sizes in .mat files and can be directly used on the test sets.**
- **calculs.ipynb**: This file is a notebook explaining the calculation of Jacobians using the *sympy* symbolic calculation library. These Jacobians are respectively stored in the **Jacobians** folder.
- **classification.py**: This file allows classification using the *pytorch* library using LSTM neural networks. The process is as follows:
    - All **.npz** files in the Data/Clean directory are retrieved and randomly divided into the training set and test set according to a certain proportion (configurable).
    - The data from these training and test sets are divided multiple times into sequences of various lengths (configurable) and sent to the neural network for classification and evaluation on the test set. 
    - During training, the accuracy curves on the training and test sets and the loss functions are updated in real time. This code automatically detects if the GPU can be used with **pytorch**. However, this code is only for informative purposes and does not give the best results we have obtained.
- **convert.py**: This file converts all **.npz** files in the **Data/Clean** directory to **.mat** files that can be used by Matlab. These files are saved in Matlab/AG, Matlab/GB, and Matlab/MD.
- **data_prep.py**: This file writes the data contained in the **.npz** files in the Data/AG, Data/GB, and Data/MD directories to a Clean folder after manual labeling.
- **etalonnage.py**: This file calibrates the sensors of the three phones used. The approach used is described in the report and the files used for calibration are stored in Data/Etalonnage.
- **Kalman_filter_tot.py**: This file contains the functions that perform the two successive Kalman filters mentioned in the report. Running this file saves the filtered data $\bm\theta$, $\dot{\bm\theta}$, and $\dot{\bm{v}}$ in an **.npz** format in the directories associated with the raw **.xls** data. Other options are available:
    - *display_traj*: If set to *True*, this option displays the trajectories over time after the extended Kalman filter is applied.
    - *display_cov*: If set to *True*, this option displays the covariance matrices $\bm P$ over time after the extended Kalman filter is applied.
- **Kalman_filtre.py**: This file has the same function as the previous file, but it only performs the extended Kalman filter on the first model. Therefore, only $\bm\theta$ and $\dot{\bm\theta}$ are filtered if this file is executed.
- **modele_pendule_pesant.py**: Running this file allows you to see an animation of the heavy pendulum. It was used to validate this model in the first part of the report.
- **print_clean.py**: Display file used for testing during labeling.
- **test_data_acce_gyr**, **test_data.py**, **test_lin_dis.py**, **test_res.py**: Other test files...
- **utils.py**: This file lists the general utility functions that can be called in other code files. ***The functions calculating gradients and discretizations mentioned in the report are included here.***