import numpy as np
import matplotlib.pyplot as plt

g = 9.81

L = 2.0
Nx = 400
dx = L / Nx
CFL = 0.5
T_final = 1.0

h0 = 1.0
x_bar = L / 4


def init_conditions():
    x = (np.arange(Nx) + 0.5) * dx
    h = np.zeros(Nx)
    u = np.zeros(Nx)

    h[x <= x_bar] = h0
    q = h * u

    U = np.vstack((h, q))
    return x, U


def hll_flux(UL, UR):
    hL, qL = UL
    hR, qR = UR

    if hL > 1e-8:
        uL = qL / hL
        cL = np.sqrt(g * hL)
    else:
        uL = 0.0
        cL = 0.0

    if hR > 1e-8:
        uR = qR / hR
        cR = np.sqrt(g * hR)
    else:
        uR = 0.0
        cR = 0.0

    sL = min(uL - cL, uR - cR)
    sR = max(uL + cL, uR + cR)

    FL = np.array([qL, qL ** 2 / hL + 0.5 * g * hL ** 2]) if hL > 1e-8 else np.zeros(2)
    FR = np.array([qR, qR ** 2 / hR + 0.5 * g * hR ** 2]) if hR > 1e-8 else np.zeros(2)

    if sL >= 0:
        return FL
    elif sR <= 0:
        return FR
    else:
        return (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL)


def compute_dt(U):
    h = U[0, :]
    q = U[1, :]

    u = np.zeros_like(h)
    mask = h > 1e-8
    u[mask] = q[mask] / h[mask]
    c = np.sqrt(g * h)

    vmax = np.max(np.abs(u) + c)
    if vmax < 1e-8:
        return 1e-3

    return CFL * dx / vmax


def apply_boundary_riviere_infinie(U, h_amont=h0):
    U[0, 0] = h_amont
    U[1, 0] = 0.0

    U[:, -1] = U[:, -2]

    return U


def step_riviere_infinie(U, dt, h_amont=h0):
    U = apply_boundary_riviere_infinie(U, h_amont)

    F_num = np.zeros((2, Nx + 1))

    for i in range(Nx + 1):
        if i == 0:
            UL = np.array([h_amont, 0.0])
            UR = U[:, 0]
        elif i == Nx:
            UL = U[:, -1]
            UR = U[:, -1]
        else:
            UL = U[:, i - 1]
            UR = U[:, i]

        F_num[:, i] = hll_flux(UL, UR)

    Unew = U.copy()
    for i in range(Nx):
        Unew[:, i] -= dt / dx * (F_num[:, i + 1] - F_num[:, i])

    return Unew


def solution_analytique_riviere_infinie(x, t, h0, x_bar):
    if t <= 0:
        h = np.where(x <= x_bar, h0, 0.0)
        u = np.zeros_like(x)
        return h, u

    c0 = np.sqrt(g * h0)
    h = np.zeros_like(x)
    u = np.zeros_like(x)

    for i, xi in enumerate(x):
        s = (xi - x_bar) / t

        if s <= -c0:
            h[i] = h0
            u[i] = 0.0
        elif -c0 < s <= 2 * c0:
            h[i] = (2 * c0 - s) ** 2 / (9 * g)
            u[i] = 2 / 3 * (c0 + s)
        else:
            h[i] = 0.0
            u[i] = 0.0

    return h, u


def run_simulation_riviere_infinie():
    x, U = init_conditions()
    h_initial = U[0, :].copy()

    h_history = [h_initial.copy()]
    t_history = [0.0]

    t = 0.0
    iteration = 0

    print("Simulation rupture de barrage avec rivière infinie à gauche")
    print("=" * 60)

    while t < T_final:
        dt = compute_dt(U)
        if t + dt > T_final:
            dt = T_final - t

        U = step_riviere_infinie(U, dt, h_amont=h0)
        t += dt
        iteration += 1

        if iteration % 20 == 0:
            h_history.append(U[0, :].copy())
            t_history.append(t)

        if iteration % 100 == 0:
            print(f"Itération {iteration:4d}, t = {t:.4f}s, dt = {dt:.2e}")

    h_final = U[0, :].copy()

    h_analytique, u_analytique = solution_analytique_riviere_infinie(x, T_final, h0, x_bar)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(x, h_initial, 'b--', label="Initiale (t=0)", alpha=0.7)
    axes[0].plot(x, h_final, 'r-', label=f"Numérique (t={T_final}s)", linewidth=2)
    axes[0].plot(x, h_analytique, 'g--', label="Analytique", linewidth=1.5, alpha=0.7)
    axes[0].fill_between(x, 0, h_final, alpha=0.2, color='red')
    axes[0].axvline(x=x_bar, color='k', linestyle=':', label="Position barrage")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("h(x,t) [m]")
    axes[0].set_title("Hauteur d'eau - Comparaison finale")
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    colors = plt.cm.plasma(np.linspace(0, 1, len(h_history)))
    for i, (h_val, t_val) in enumerate(zip(h_history, t_history)):
        if i % 2 == 0:
            axes[1].plot(x, h_val, color=colors[i], alpha=0.7,
                         label=f"t={t_val:.2f}s" if i % 5 == 0 else "")
    axes[1].axvline(x=x_bar, color='k', linestyle=':', linewidth=2)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("h(x,t) [m]")
    axes[1].set_title("Évolution temporelle")
    axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Rupture de barrage avec rivière infinie à gauche (t={T_final}s)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_simulation_riviere_infinie()