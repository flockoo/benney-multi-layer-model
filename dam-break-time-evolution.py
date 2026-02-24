import numpy as np
import matplotlib.pyplot as plt

g = 9.81

L = 10.0
Nx = 1000
dx = L / Nx
CFL = 0.5
T_final = 5.0
h0 = 1.0
x_bar = 2.0


def init_conditions(Nx, L, x_bar, h0):
    dx = L / Nx
    x = (np.arange(Nx) + 0.5) * dx
    h = np.zeros(Nx)
    u = np.zeros(Nx)

    h[x <= x_bar] = h0
    q = h * u

    U = np.vstack((h, q))
    return x, U


def flux(U):
    h = U[0, :]
    q = U[1, :]

    F = np.zeros_like(U)
    F[0, :] = q

    mask = h > 1e-8
    F[1, mask] = q[mask] ** 2 / h[mask] + 0.5 * g * h[mask] ** 2
    F[1, ~mask] = 0.0

    return F


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

    FL = flux(UL.reshape(2, 1))[:, 0]
    FR = flux(UR.reshape(2, 1))[:, 0]

    if sL >= 0:
        return FL
    elif sR <= 0:
        return FR
    else:
        return (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL)


def compute_dt(U, dx):
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


def apply_boundary(U):
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]
    return U


def step(U, dt, dx):
    U = apply_boundary(U)
    Nx = U.shape[1]

    F_num = np.zeros((2, Nx + 1))

    for i in range(Nx + 1):
        if i == 0:
            UL = U[:, 0]
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


def solution_analytique_ritter(x, t, h0, x0):
    if t <= 0:
        h = np.where(x <= x0, h0, 0.0)
        return h

    c0 = np.sqrt(g * h0)
    h = np.zeros_like(x)

    front_position = x0 + 2 * c0 * t

    for i, xi in enumerate(x):
        s = (xi - x0) / t

        if s <= -c0:
            h[i] = h0
        elif -c0 < s <= 2 * c0:
            h[i] = (2 * c0 - s) ** 2 / (9 * g)
        else:
            h[i] = 0.0

    return h


def simulation_multi_temps(Nx, L, x_bar, h0, T_final, temps_a_sauvegarder):
    dx = L / Nx
    x, U = init_conditions(Nx, L, x_bar, h0)

    temps_a_sauvegarder = sorted(temps_a_sauvegarder)
    if T_final not in temps_a_sauvegarder:
        temps_a_sauvegarder.append(T_final)

    solutions = {}

    h_init = U[0, :].copy()
    solutions[0.0] = {'h': h_init, 'x': x.copy()}

    t = 0.0
    next_save_idx = 0

    print(f"\nDébut de la simulation jusqu'à t = {T_final}s...")
    print(f"Temps à sauvegarder : {temps_a_sauvegarder}")
    print("=" * 60)

    iteration = 0

    while t < T_final and next_save_idx < len(temps_a_sauvegarder):
        dt = compute_dt(U, dx)

        next_t = temps_a_sauvegarder[next_save_idx]
        if t + dt > next_t:
            dt = next_t - t

        if t + dt > T_final:
            dt = T_final - t

        U = step(U, dt, dx)
        t += dt

        if abs(t - next_t) < 1e-10 or t >= next_t:
            h_t = U[0, :].copy()
            solutions[next_t] = {'h': h_t, 'x': x.copy()}
            print(f"✓ t = {next_t:.2f}s sauvegardé (itération {iteration})")
            next_save_idx += 1

        if iteration % 1000 == 0 and iteration > 0:
            print(f"  Progression: t = {t:.2f}/{T_final}s, dt = {dt:.4f}")

        iteration += 1

    print("=" * 60)
    print(f"Simulation terminée ! {iteration} itérations")
    print(f"dt moyen = {T_final / iteration:.4f}s")

    return solutions


def plot_solutions_multi_temps(solutions, h0, x_bar, L, T_final):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    temps = sorted([t for t in solutions.keys() if t > 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(temps) + 1))

    sol_t0 = solutions[0.0]
    ax.plot(sol_t0['x'], sol_t0['h'], 'k--', linewidth=2, label='t = 0 (initial)')

    for i, t in enumerate(temps):
        sol = solutions[t]
        ax.plot(sol['x'], sol['h'], color=colors[i + 1], linewidth=2,
                label=f't = {t:.2f}s (num)')

        if t <= 2 * np.sqrt(g * h0) and t > 0:
            h_ana = solution_analytique_ritter(sol['x'], t, h0, x_bar)
            ax.plot(sol['x'], h_ana, color=colors[i + 1], linestyle=':',
                    linewidth=1.5, alpha=0.7)

    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('Hauteur h [m]', fontsize=12)
    ax.set_title(f'Rupture de barrage - Évolution de la hauteur (t_max = {T_final}s)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.set_xlim([0, L])
    ax.set_ylim([0, h0 * 1.1])
    ax.axvline(x=x_bar, color='red', linestyle='--', alpha=0.5, label='Barrage')

    plt.suptitle(f'Solution à différents temps - Domaine L={L}m, Nx={Nx}, CFL={CFL}', fontsize=16)
    plt.tight_layout()
    plt.savefig('rupture_barrage_longue_duree.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_evolution_temporelle(solutions, x_point, L):
    x = solutions[0.0]['x']
    idx = np.argmin(np.abs(x - x_point))
    x_reel = x[idx]

    temps = sorted(solutions.keys())
    h_evol = []

    for t in temps:
        sol = solutions[t]
        h_evol.append(sol['h'][idx])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(temps, h_evol, 'b-', linewidth=2)
    ax.set_xlabel('Temps [s]', fontsize=12)
    ax.set_ylabel('Hauteur h [m]', fontsize=12)
    ax.set_title(f'Évolution temporelle en x = {x_reel:.2f}m', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evolution_temporelle_point.png', dpi=150, bbox_inches='tight')
    plt.show()


def comparaison_durees():
    durees = [1.0, 2.0, 5.0, 10.0]
    temps_communs = [0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, duree in enumerate(durees):
        temps_save = [t for t in temps_communs if t <= duree]

        print(f"\n--- Simulation pour T_final = {duree}s ---")
        solutions = simulation_multi_temps(Nx, L, x_bar, h0, duree, temps_save)

        ax = axes[idx]

        temps_plots = sorted([t for t in solutions.keys() if t > 0])
        colors = plt.cm.plasma(np.linspace(0, 1, len(temps_plots)))

        for j, t in enumerate(temps_plots):
            sol = solutions[t]
            ax.plot(sol['x'], sol['h'], color=colors[j], linewidth=2,
                    label=f't={t:.1f}s')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('h [m]')
        ax.set_title(f'T_final = {duree}s')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.axvline(x=x_bar, color='red', linestyle='--', alpha=0.5)
        ax.set_xlim([0, L])
        ax.set_ylim([0, h0 * 1.1])

    plt.suptitle(f'Comparaison pour différentes durées de simulation', fontsize=16)
    plt.tight_layout()
    plt.savefig('comparaison_durees.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SIMULATION RUPTURE DE BARRAGE ")
    print("=" * 60)
    print(f"Domaine: [0, {L}] m")
    print(f"Barrage à x = {x_bar} m")
    print(f"Hauteur initiale: h0 = {h0} m")
    print(f"Temps final: T_final = {T_final} s")
    print(f"Mailles: Nx = {Nx} (dx = {L / Nx:.3f} m)")
    print(f"CFL = {CFL}")

    if T_final <= 2.0:
        temps_voulus = [0.2, 0.5, 1.0, 1.5, 2.0]
    elif T_final <= 5.0:
        temps_voulus = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    else:
        temps_voulus = [1.0, 2.0, 5.0, 8.0, 10.0]

    solutions = simulation_multi_temps(Nx, L, x_bar, h0, T_final, temps_voulus)

    plot_solutions_multi_temps(solutions, h0, x_bar, L, T_final)