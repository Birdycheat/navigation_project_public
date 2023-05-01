import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import display_np_angles, drop_first_and_last_np, np_save, load_to_array, split_array
import os



# By hand preparation

folder_save = "Data/Clean/AG/"
folder = "Data/AG/"

file_name = "Bike1.npz"
data = load_to_array(folder + file_name)
np_bike1 = drop_first_and_last_np(data, 8, 30.3)
np_bike2 = drop_first_and_last_np(data, 44, 156)
np_bike3 = drop_first_and_last_np(data, 185, 200)
np_bike4 = drop_first_and_last_np(data, 205, 276)
np_save(np_bike1, folder_save + "bike_1.npz")
np_save(np_bike2, folder_save + "bike_2.npz")
np_save(np_bike3, folder_save + "bike_3.npz")
np_save(np_bike4, folder_save + "bike_4.npz")

file_name = "Stairs1.npz"
data = load_to_array(folder + file_name)
np_stair1 = drop_first_and_last_np(data, 1.70, 15)
np_save(np_stair1, folder_save + "stairs_1.npz")

file_name = "Stairs2.npz"
data = load_to_array(folder + file_name)
np_stair2 = drop_first_and_last_np(data, 3.5, 14)
np_save(np_stair2, folder_save + "stairs_2.npz")

file_name = "Stairs3.npz"
data = load_to_array(folder + file_name)
np_stair3 = drop_first_and_last_np(data, 2, 12.5)
np_save(np_stair3, folder_save + "stairs_3.npz")

file_name = "Stairs4.npz"
data = load_to_array(folder + file_name)
np_stair4 = drop_first_and_last_np(data, 3.7, 14)
np_save(np_stair4, folder_save + "stairs_4.npz")

file_name = "Stairs5.npz"
data = load_to_array(folder + file_name)
np_stairs5 = drop_first_and_last_np(data, 3.78, 10.3)
np_stairs6 = drop_first_and_last_np(data, 14.2, 16.6)
np_stairs7 = drop_first_and_last_np(data, 19.24, 28.22)
np_save(np_stairs5, folder_save + "stairs_5.npz")
np_save(np_stairs6, folder_save + "stairs_6.npz")
np_save(np_stairs7, folder_save + "stairs_7.npz")

file_name = "Stairs6.npz"
data = load_to_array(folder + file_name)
np_stairs8 = drop_first_and_last_np(data, 3.6, 14.68)
np_stairs9 = drop_first_and_last_np(data, 20.66, 31)
np_stairs10 = drop_first_and_last_np(data, 36.96, 49.07)
np_save(np_stairs8, folder_save + "stairs_8.npz")
np_save(np_stairs9, folder_save + "stairs_9.npz")
np_save(np_stairs10, folder_save + "stairs_10.npz")



file_name = "Walk.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 4, 60)
np_gyr_walk1, np_gyr_tmp = split_array(data, 24.9)
np_gyr_ml1, np_gyr_walk2 = split_array(np_gyr_tmp, 27.7)
np_save(np_gyr_walk1, folder_save + "walk_1.npz")
np_save(np_gyr_walk2, folder_save + "walk_2.npz")

file_name = "Walk2.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 4, 155)
np_save(data, folder_save + "walk_3.npz")

folder_save = "Data/Clean/GB/"
folder = "Data/GB/"
file_name = "StairUp1.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 4, 41.8)
np_gyr_ml1, np_gyr_tmp = split_array(data, 10)
np_gyr_stairs1, np_gyr_ml2 = split_array(np_gyr_tmp, 36)
np_save(np_gyr_stairs1, folder_save + "stairs_1.npz")

file_name = "StairUp2.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 4.4, 41.7)
np_gyr_ml3, np_gyr_tmp = split_array(data, 11.1)
np_gyr_stairs1, np_gyr_ml4 = split_array(np_gyr_tmp, 37)
np_save(np_gyr_stairs1, folder_save + "stairs_2.npz")

file_name = "StairUp3.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 4.4, 40)
np_gyr_ml3, np_gyr_tmp = split_array(data, 10)
np_gyr_stairs1, np_gyr_ml4 = split_array(np_gyr_tmp, 36.5)
np_save(np_gyr_stairs1, folder_save + "stairs_3.npz")

file_name = "Walk1.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 4, 138)
np_save(data, folder_save + "walk_1.npz")

file_name = "Walk2.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 3.8, 232)
np_save(data, folder_save + "walk_2.npz")

folder_save = "Data/Clean/MD/"
folder = "Data/MD/"
file_name = "stairs_25-03_1.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 2.8, 15.2)
np_save(data, folder_save + "stairs_1.npz")

file_name = "stairs_25-03_coatpocketwithouthand_1.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 2, 15.71)
np_save(data, folder_save + "stairs_2.npz")

file_name = "stairs_25-03_coatpocketwithouthand_2.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 2, 12.5)
np_save(data, folder_save + "stairs_3.npz")

file_name = "stairs_up_27-04_1.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 2, 14.42)
np_save(data, folder_save + "stairs_4.npz")

file_name = "stairs_up_27-04_3 (1bis).npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 2.2, 14)
np_save(data, folder_save + "stairs_5.npz")

file_name = "stairs_up_backpocket_27-04_2.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 3, 11)
np_save(data, folder_save + "stairs_6.npz")

file_name = "stairs_up_backpocket_2by2_27-04_4.npz"
data = load_to_array(folder + file_name)
data = drop_first_and_last_np(data, 2.33, 9.51)
np_save(data, folder_save + "stairs_7.npz")
         
