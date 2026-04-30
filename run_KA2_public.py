# DiscreteLatticeEM_v1.2 Public
# Lorentz-like Dynamics in 2D Chiral Cellular Automata
# Author: P.K., 2026 | License: MIT
# Repo: https://github.com/pattologos12-glitch/DiscreteLatticeEM

import numpy as np
import matplotlib.pyplot as plt

N = 100
k_coulomb = 0.048
K_base = 10.0
alpha = 0.5
beta_L = 0.15
beta_R = 0.10
damping = 0.99999
pump = 0.005
advekce = 0.02
steps = 200
B_fields = [0.0, 0.1]

def init_vortex_2d(N, center, Q_sign, sigma=3.0):
    x = np.arange(N); y = np.arange(N)
    X, Y = np.meshgrid(x, y)
    dx = X - center[0]; dy = Y - center[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    amp = 15.0 * np.exp(-r**2 / (2*sigma**2))
    phase = Q_sign * theta
    return amp * np.cos(phase)

def evolve_lattice(phi, K_sat, B_field, vx_prev):
    new_phi = np.copy(phi)
    for i in range(1, N-1):
        for j in range(1, N-1):
            if abs(phi[i,j]) > K_sat[i,j]:
                excess = abs(phi[i,j]) - K_sat[i,j]
                new_phi[i,j] = np.sign(phi[i,j]) * K_sat[i,j] * alpha
                new_phi[i, j-1] += np.sign(phi[i,j]) * excess * beta_L
                new_phi[i, j+1] += np.sign(phi[i,j]) * excess * beta_R
                new_phi[i-1, j] += np.sign(phi[i,j]) * excess * 0.125
                new_phi[i+1, j] += np.sign(phi[i,j]) * excess * 0.125
    Q = +1
    F_lorentz_y = k_coulomb * Q * vx_prev * B_field
    shift = int(np.round(F_lorentz_y * 100))
    if shift!= 0:
        new_phi = np.roll(new_phi, shift, axis=0)
    adv_phi = np.copy(new_phi)
    for i in range(N):
        for j in range(1, N):
            adv_phi[i,j] = new_phi[i,j]*(1-advekce) + new_phi[i,j-1]*advekce
    return adv_phi * damping + pump

def run_sim(B_field):
    x = np.arange(N); y = np.arange(N)
    X, Y = np.meshgrid(x, y)
    K_sat = K_base * (1 - 0.2*X/N) * (1 + B_field * Y/N)
    phi = init_vortex_2d(N, [20, 50], Q_sign=+1)
    trajectory, energy = [], []
    vx_prev = 0.03
    for step in range(steps):
        phi = evolve_lattice(phi, K_sat, B_field, vx_prev)
        E = np.sum(phi**2)
        com_y, com_x = np.unravel_index(np.argmax(phi**2), phi.shape) if E > 1 else [50,20]
        if len(trajectory) > 0: vx_prev = com_x - trajectory[-1][0]
        trajectory.append([com_x, com_y]); energy.append(E)
    return np.array(trajectory), np.array(energy)

results = {}
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for B in B_fields:
    traj, E = run_sim(B)
    results[B] = {'traj': traj, 'energy': E}
    dx = traj[-1,0] - traj[0,0]; dy = traj[-1,1] - traj[0,1]
    V_eff = dy / dx if dx > 0 else 0
    print(f"B={B:.1f} | dX={dx:.1f} | dY={dy:.1f} | V_eff={V_eff:.3f} | E_end={E[-1]:.1f}")
    ax[0].plot(traj[:,0], traj[:,1], label=f'B={B}, V_eff={V_eff:.2f}')
    ax[1].plot(E, label=f'B={B}')

ax[0].set_title('Trajectory'); ax[0].set_xlabel('X'); ax[0].set_ylabel('Y'); ax[0].legend(); ax[0].grid()
ax[1].set_title('Energy'); ax[1].set_xlabel('Steps'); ax[1].set_ylabel('E'); ax[1].legend(); ax[1].grid()
plt.tight_layout(); plt.savefig('trajectory_energy.png'); plt.show()