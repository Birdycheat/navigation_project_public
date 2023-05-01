import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import display_np
import os

avoid = ['Etalonnage']
directory = "Data/Clean"
for folder in os.listdir(directory):
    if folder not in avoid:
        for filename in os.listdir(directory+"/"+folder):
            entire_path = directory+"/"+folder+"/"+filename
            extension = filename.split(".")[-1]
            if extension == "npz":
                print(entire_path)
                data = np.load(entire_path)
                np_gyr = np.block([[data.f.t], [data.f.x[3:6, :]]])
                np_acc = np.block([[data.f.t], [data.f.x[6:, :]]])
                display_np(np_acc, np_gyr)
            