from utils import jacobian_gyro, jacobian_gyro_cmd, discretize_state
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# constants
L = 0.25*1.8 # 25% of 1.80m
M = 0.04*80 # 4% of 80kg
m = 0.14*80 # 14% of 80kg
r = 0.1
I_xx = (1/4*m*r**2+1/3*m*L**2)
I_yy = (1/4*m*r**2+1/3*m*L**2)
I_zz = 1/2*m*r**2

# Montreal estimation of gravity vector
g = 9.806


# state function
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


# conditions initiales
x0 = [1.67, 0.0, 0.0, 0.1, 0.001, 0.0001]

# temps d'intégration
t_span = (0.0, 20.0)

# entrées pour Cm_x, Cm_y, Cm_z
Cm_x = lambda t: 1*np.cos(6*t)*np.exp(-100*t)
Cm_y = lambda t: 1*np.cos(6*0.8*t)*np.exp(-10*t)
Cm_z = lambda t: 1*np.cos(6*0.8*t)*np.exp(-10*t)

# résolution de l'équation différentielle
dt = 0.001
N = round((t_span[1]-t_span[0])/dt)
ts = np.linspace(t_span[0], t_span[1], N)
sol = solve_ivp(f, t_span, x0, args=(Cm_x, Cm_y, Cm_z), t_eval=ts)

Nlin = 50
x_lin = np.zeros((6, Nlin*(N-1) - N + 2))
x_lin_dis = np.zeros((6, Nlin*(N-1) - N + 2))
x_lin[:, 0] = x0
x_lin_dis[:, 0] = x0
u = np.block([[Cm_x(ts)], [Cm_y(ts)], [Cm_z(ts)]])
index = 0
ts_glob_lin = np.linspace(t_span[0], t_span[1], Nlin*(N-1) - N + 2)


for i in range(ts.size-1):
    # Linearization in a neighborhood of the current working point
    A = jacobian_gyro(x_lin[:, i], u[:, i], L, M, m, I_xx, I_yy, I_zz, g)
    B = jacobian_gyro_cmd(x_lin[:, i], u[:, i], L, M, m, I_xx, I_yy, I_zz, g)
    x0 = np.copy(x_lin[:, index])
    ts_lin = np.linspace(ts[i], ts[i+1], Nlin)
    df = lambda t, x, Cm_x, Cm_y, Cm_z: f(ts[i], x0, Cm_x, Cm_y, Cm_z) + A@(x-x0) + B@np.array([Cm_x(t) - Cm_x(ts[i]), Cm_y(t) - Cm_y(ts[i]), Cm_z(t) - Cm_z(ts[i])])
    sol_tmp = solve_ivp(df, (ts[i], ts[i+1]), y0 = x0, args=(Cm_x, Cm_y, Cm_z), t_eval=ts_lin)
    x_lin[:,index:index + Nlin] = sol_tmp.y
    # Discretization
    Ak, Bk, _ = discretize_state(A, B, None, ts_glob_lin[0], ts_glob_lin[1])
    vec_Ck = (f(ts[i], x0, Cm_x, Cm_y, Cm_z) - (A@x0 + B@np.array([Cm_x(ts[i]), Cm_y(ts[i]), Cm_z(ts[i])]))).reshape((6, 1))
    _, Ck, _ = discretize_state(A, vec_Ck, None, ts_glob_lin[0], ts_glob_lin[1])
    for j in range(index+1, index + Nlin):
        x_lin_dis[:, j] = Ck.reshape((6,)) + Ak@x_lin_dis[:, j-1] + Bk@np.array([Cm_x(ts_glob_lin[j-1]), Cm_y(ts_glob_lin[j-1]), Cm_z(ts_glob_lin[j-1])])
    index += Nlin-1
    

# display of the results
fig, axs = plt.subplots(3, 2, figsize=(10, 10)) 
axs[0, 0].plot(sol.t, sol.y[0])
axs[0, 0].plot(ts_glob_lin, x_lin[0,:], label = "linearized model")
axs[0, 0].plot(ts_glob_lin, x_lin_dis[0,:], label = "discretized model")
axs[0, 0].legend()
axs[0, 0].set_ylabel('$x_1$')
axs[0, 1].plot(sol.t, sol.y[1])
axs[0, 1].plot(ts_glob_lin, x_lin[1,:], label = "linearized model")
axs[0, 1].plot(ts_glob_lin, x_lin_dis[1,:], label = "discretized model")
axs[0, 1].legend()
axs[0, 1].set_ylabel('$x_2$')
axs[1, 0].plot(sol.t, sol.y[2])
axs[1, 0].plot(ts_glob_lin, x_lin[2,:], label = "linearized model")
axs[1, 0].plot(ts_glob_lin, x_lin_dis[2,:], label = "discretized model")
axs[1, 0].legend()
axs[1, 0].set_ylabel('$x_3$')
axs[1, 1].plot(sol.t, sol.y[3])
axs[1, 1].plot(ts_glob_lin, x_lin[3,:], label = "linearized model")
axs[1, 1].plot(ts_glob_lin, x_lin_dis[3,:], label = "discretized model")
axs[1, 1].legend()
axs[1, 1].set_ylabel('$x_4$')
axs[2, 0].plot(sol.t, sol.y[4])
axs[2, 0].plot(ts_glob_lin, x_lin[4,:], label = "linearized model")
axs[2, 0].plot(ts_glob_lin, x_lin_dis[4,:], label = "discretized model")
axs[2, 0].legend()
axs[2, 0].set_ylabel('$x_5$')
axs[2, 1].plot(sol.t, sol.y[5])
axs[2, 1].plot(ts_glob_lin, x_lin[5,:], label = "linearized model")
axs[2, 1].plot(ts_glob_lin, x_lin_dis[5,:], label = "discretized model")
axs[2, 1].legend()
axs[2, 1].set_ylabel('$x_6$')
plt.show()
