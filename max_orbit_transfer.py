import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


def dynamics(t, states):
    r, u, v, lr, lu, lv = states

    sinphi = -lu/np.sqrt(lu**2 + lv**2)
    cosphi = -lv/np.sqrt(lu**2 + lv**2)

    r_dot = u
    u_dot = (v**2/r) - mu/(r**2) + T*sinphi/(m0 + m1*t)
    v_dot = -(u*v)/r + T*cosphi/(m0 + m1*t)
    lr_dot = -lu * ((-v**2/r**2) + ((2*mu)/r**3) - lv * ((u*v)/r**2))
    lu_dot = -lr + lv * (v/r)
    lv_dot = -lu * 2 * (v/r) + lv * (u/r)

    return np.vstack((r_dot, u_dot, v_dot, lr_dot, lu_dot, lv_dot))


def bounds(ya, yb):
    ri, ui, vi, lri, lui, lvi = ya
    rf, uf, vf, lrf, luf, lvf = yb
    return np.array([ri - 1., ui, vi-np.sqrt(mu/ri), uf, vf - np.sqrt(mu/rf), lrf + 1 - lvf*np.sqrt(mu)/2/(rf**(3/2))])


if __name__ == "__main__":
    mu = 1.
    m0 = 1.
    m1 = -.07485
    T = .1405
    NUM_STATES = 6
    N = 100
    tf = 4.
    x = np.linspace(0, tf, N)

    # Initial Guess
    y_a = np.zeros((NUM_STATES, x.size))
    y_a[0, :] = 1.
    y_a[1, :] = 0.
    y_a[2, :] = 1.
    y_a[3, :] = -1.
    y_a[4, :] = -1.
    y_a[5, :] = -1.

    res_a = solve_bvp(dynamics, bounds, x, y_a)

    x_plot = np.linspace(0, tf, res_a.y.shape[1])

    r_states = res_a.y[0, :]
    u_states = res_a.y[1, :]
    v_states = res_a.y[2, :]

    lr_states = res_a.y[3, :]
    lu_states = res_a.y[4, :]
    lv_states = res_a.y[5, :]

    plt.plot(x_plot, r_states, 'b', label='r')
    plt.plot(x_plot, u_states, 'g', label='u')
    plt.plot(x_plot, v_states, 'r', label='v')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.plot(x_plot, lr_states, 'b', label='lr')
    plt.plot(x_plot, lu_states, 'g', label='lu')
    plt.plot(x_plot, lv_states, 'r', label='lv')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    init_rad = 1.
    fin_rad = r_states[-1]
    fig, ax = plt.subplots()
    ax.set(xlim=(-2., 2.), ylim=(0., 3.))
    init_orbit = plt.Circle((0, 0), init_rad, color='b', fill=False)
    final_orbit = plt.Circle((0, 0), fin_rad, color='r', fill=False)
    ax.add_artist(init_orbit)
    ax.add_artist(final_orbit)
    plt.show()