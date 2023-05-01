import pandas as pd
import matplotlib.pyplot as plt

dfs = pd.read_excel("Data/AG/ex1gyro.xls")
print(dfs.head(10))

plt.subplots(3, 1)

plt.subplot(3, 1, 1)
plt.plot(dfs["Time (s)"], dfs["Gyroscope x (rad/s)"], color="yellow")
plt.ylabel("Gyroscope x (rad/s)")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(dfs["Time (s)"], dfs["Gyroscope y (rad/s)"], color="red")
plt.ylabel("Gyroscope y (rad/s)")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(dfs["Time (s)"], dfs["Gyroscope z (rad/s)"], color="black")
plt.xlabel("Time (s)")
plt.ylabel("Gyroscope z (rad/s)")
plt.grid(True)


plt.show()
