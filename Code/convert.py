from scipy.io import savemat
import numpy as np
import os



folder_input = "Data/Clean/"
folder_output = "Code/Matlab/"
names = ["AG", "GB", "MD"]

for name in names:
    directory = folder_output + name
    for root, dirs, files in os.walk(directory):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
            
            
indexes_selected_measurements = [0, 1, 2, 3, 4, 5, 6, 7, 8]
for name in names:
    folder_npz = folder_input + name
    for filename in os.listdir(folder_npz):
        if filename.split('.')[-1] == "npz":
            file_path = folder_npz + "/" + filename
            d = np.load(file_path)
            fm = folder_output + name + '/' + filename.split('.')[0] + '.mat'
            savemat(fm, {"x" : d.f.x[indexes_selected_measurements], "y" : d.f.t})
            print('generated ', fm, 'from', filename)