import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# cst
L = 0.25*1.8 # 25% of 1.80m
M = 0.04*80 # 4% of 80kg
m = 0.14*80 # 14% of 80kg
r = 0.1
I_xx = (1/4*m*r**2+1/3*m*L**2)
I_yy = (1/4*m*r**2+1/3*m*L**2)
I_zz = 1/2*m*r**2

# Montreal estimation of gravity vector
g = 9.806

# state function for the first EKF
def f(t, x, Cm_x, Cm_y, Cm_z):
    x1, x2, x3, x4, x5, x6 = x
    f1 = x4
    f2 = x5
    f3 = x6
    f4 = (2*L**2*np.cos(x2)*np.sin(x2)*x4*x5 + 
          (M+m/2)*L*g*np.cos(x2)*np.sin(x1) + Cm_x(t)) / (I_xx + L**2*np.cos(x2)**2*(M+m/4))
    f5 = ((M+m/2)*L*g*np.cos(x1)*np.sin(x2) + Cm_y(t)) / (I_yy + L**2*(M+m/4))
    f6 = Cm_z(t) / I_zz
    return [f1, f2, f3, f4, f5, f6]

# Rotations matrices with respect to the 3 principal axes
def Rx(angle):
    return np.block([[[np.ones((angle.size))],[np.zeros(angle.size)],[np.zeros(angle.size)]],[[np.zeros(angle.size)],[np.cos(angle)],[-np.sin(angle)]],[[np.zeros(angle.size)],[np.sin(angle)],[np.cos(angle)]]])
def Ry(angle):
    return np.block([[[np.cos(angle)],[np.zeros(angle.size)],[np.sin(angle)]],[[np.zeros(angle.size)],[np.ones((angle.size))],[np.zeros(angle.size)]],[[-np.sin(angle)],[np.zeros(angle.size)],[np.cos(angle)]]])
def Rz(angle):
    return np.block([[[np.cos(angle)],[-np.sin(angle)],[np.zeros(angle.size)]],[[np.sin(angle)],[np.cos(angle)],[np.zeros(angle.size)]],[[np.zeros(angle.size)],[np.zeros(angle.size)],[np.ones((angle.size))]]])

def R_hip_rod(angles):
    return np.einsum("ipk, pjk -> ijk", np.einsum("ipk, pjk -> ijk", Rx(angles[0]), Ry(angles[1])), Rz(angles[2]))

def R_rod_sens():
    return np.array([[-1,0,0],[0,0,-1],[0,-1,0]])

    

# initial conditions
x0 = [1.67, 0.0, 0.0, 0.1, 0.001, 0.0001]

# integration times
t_span = (0.0, 20.0)


# Torques inputs
Cm_x = lambda t: 0.0*np.cos(6*t)*np.exp(-100*t)
Cm_y = lambda t: 0.0*np.cos(6*0.8*t)*np.exp(-10*t)
Cm_z = lambda t: 0.0*np.cos(6*0.8*t)*np.exp(-10*t)

# Computation of the solution of the system
dt = 0.01
N = round((t_span[1]-t_span[0])/dt)
ts = np.linspace(t_span[0], t_span[1], N)
sol = solve_ivp(f, t_span, x0, args=(Cm_x, Cm_y, Cm_z), t_eval=ts)

# display the results
fig, axs = plt.subplots(3, 2, figsize=(10, 10)) 
axs[0, 0].plot(sol.t, sol.y[0])
axs[0, 0].set_ylabel('$x_1$')
axs[0, 1].plot(sol.t, sol.y[1])
axs[0, 1].set_ylabel('$x_2$')
axs[1, 0].plot(sol.t, sol.y[2])
axs[1, 0].set_ylabel('$x_3$')
axs[1, 1].plot(sol.t, sol.y[3])
axs[1, 1].set_ylabel('$x_4$')
axs[2, 0].plot(sol.t, sol.y[4])
axs[2, 0].set_ylabel('$x_5$')
axs[2, 1].plot(sol.t, sol.y[5])
axs[2, 1].set_ylabel('$x_6$')
plt.show()


# 3D animation
fig3D = plt.figure()
ax3D = fig3D.add_subplot(projection='3d')


def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])

xpypzp = np.block([[[np.zeros((sol.y[0].size))]],[[np.zeros((sol.y[0].size))]],[[np.ones((sol.y[0].size))*L]]])+np.einsum("ipk, pj -> ijk", R_hip_rod(np.block([[sol.y[0]],[sol.y[1]],[sol.y[2]]])), np.array([[0],[0],[L]]))

xp = xpypzp[0, :, :]
yp = xpypzp[1, :, :]
zp = xpypzp[2, :, :]

data = np.array(list([xp[0, :], yp[0, :], zp[0, :]]))

line, = ax3D.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

# Setting the axes properties
ax3D.set_xlim3d([min(data[0]), max(data[0])])
ax3D.set_xlabel('X')

ax3D.set_ylim3d([min(data[1]), max(data[1])])
ax3D.set_ylabel('Y')

ax3D.set_zlim3d([min(data[2]), max(data[2])])
ax3D.set_zlabel('Z')

ani = animation.FuncAnimation(fig3D, update, N, fargs=(data, line), interval=10000/N, blit=False)
plt.show()

# display the trajectories xp, yp, zp
figp, axsp = plt.subplots(3, 1, figsize=(10, 10)) 
axsp[0].plot(sol.t, data[0])
axsp[0].set_ylabel('$x_p$')
axsp[1].plot(sol.t, data[1])
axsp[1].set_ylabel('$y_p$')
axsp[2].plot(sol.t, data[2])
axsp[2].set_ylabel('$z_p$')

plt.show()