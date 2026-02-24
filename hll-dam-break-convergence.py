import numpy as np
import matplotlib.pyplot as plt

g = 9.81

L = 1.0
CFL = 0.5
T_final = 0.2
h0 = 1.0
x_bar = 0.5


def init_conditions(Nx):
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
        u = np.zeros_like(x)
        return h, u

    c0 = np.sqrt(g * h0)
    h = np.zeros_like(x)
    u = np.zeros_like(x)

    for i, xi in enumerate(x):
        s = (xi - x0) / t

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


def compute_L2_error(u_num, u_ana, dx):
    mask = np.isfinite(u_num) & np.isfinite(u_ana)

    if np.sum(mask) == 0:
        return 0.0

    diff = u_num[mask] - u_ana[mask]
    L2_error = np.sqrt(np.sum(diff ** 2) * dx)

    return L2_error


def run_simulation(Nx):
    dx = L / Nx
    x, U = init_conditions(Nx)

    t = 0.0
    iteration = 0

    while t < T_final:
        dt = compute_dt(U, dx)
        if t + dt > T_final:
            dt = T_final - t
        U = step(U, dt, dx)
        t += dt
        iteration += 1

    h_final = U[0, :].copy()

    h_analytique, u_analytique = solution_analytique_ritter(x, T_final, h0, x_bar)

    L2_error_h = compute_L2_error(h_final, h_analytique, dx)

    u_numeric = np.zeros_like(h_final)
    mask = h_final > 1e-8
    u_numeric[mask] = U[1, mask] / h_final[mask]
    L2_error_u = compute_L2_error(u_numeric, u_analytique, dx)

    return x, h_final, h_analytique, L2_error_h, L2_error_u


def etude_convergence():
    Nx_list = [100, 200, 400, 800, 1600, 3200]

    erreurs_h = []
    erreurs_u = []
    dx_list = []

    print("=" * 70)
    print("ÉTUDE DE CONVERGENCE SPATIALE")
    print("=" * 70)
    print(f"Temps final: {T_final} s, CFL = {CFL}")
    print("-" * 70)
    print(f"{'Nx':<8} {'dx':<12} {'Erreur L2 (h)':<20} {'Erreur L2 (u)':<20}")
    print("-" * 70)

    for Nx in Nx_list:
        dx = L / Nx
        dx_list.append(dx)

        _, _, _, err_h, err_u = run_simulation(Nx)

        erreurs_h.append(err_h)
        erreurs_u.append(err_u)

        print(f"{Nx:<8} {dx:<12.6f} {err_h:<20.6e} {err_u:<20.6e}")

    print("-" * 70)

    print("\n" + "=" * 70)
    print("ORDRES DE CONVERGENCE")
    print("=" * 70)
    print(f"{'Maillages':<20} {'Ordre (h)':<20} {'Ordre (u)':<20}")
    print("-" * 70)

    ordres_h = []
    ordres_u = []

    for i in range(len(Nx_list) - 1):
        Nx1, Nx2 = Nx_list[i], Nx_list[i + 1]
        rapport_maillage = Nx2 / Nx1

        ordre_h = np.log(erreurs_h[i] / erreurs_h[i + 1]) / np.log(rapport_maillage)
        ordres_h.append(ordre_h)

        ordre_u = np.log(erreurs_u[i] / erreurs_u[i + 1]) / np.log(rapport_maillage)
        ordres_u.append(ordre_u)

        print(f"{Nx1} → {Nx2:<12} {ordre_h:<20.3f} {ordre_u:<20.3f}")

    print("-" * 70)
    print(f"{'Moyenne':<20} {np.mean(ordres_h):<20.3f} {np.mean(ordres_u):<20.3f}")
    print("=" * 70)

    return Nx_list, dx_list, erreurs_h, erreurs_u, ordres_h, ordres_u


def plot_convergence(Nx_list, dx_list, erreurs_h, erreurs_u, ordres_h, ordres_u):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.loglog(dx_list, erreurs_h, 'bo-', linewidth=2, markersize=8, label='Hauteur h')
    ax1.loglog(dx_list, erreurs_u, 'rs-', linewidth=2, markersize=8, label='Vitesse u')

    dx_ref = np.array(dx_list)
    err_ref_1 = erreurs_h[0] * (dx_ref / dx_ref[0])
    ax1.loglog(dx_ref, err_ref_1, 'k--', linewidth=1.5, label='Ordre 1 (référence)')

    ax1.set_xlabel('dx (m)', fontsize=12)
    ax1.set_ylabel('Erreur L2', fontsize=12)
    ax1.set_title('Convergence spatiale - Échelle log-log', fontsize=14)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(loc='best')

    for i, (dx, err_h, err_u) in enumerate(zip(dx_list, erreurs_h, erreurs_u)):
        ax1.annotate(f'Nx={Nx_list[i]}', (dx, err_h),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    x_pos = np.arange(len(ordres_h))
    largeur = 0.35

    ax2.bar(x_pos - largeur / 2, ordres_h, largeur, label='Ordre (h)', color='blue', alpha=0.7)
    ax2.bar(x_pos + largeur / 2, ordres_u, largeur, label='Ordre (u)', color='red', alpha=0.7)

    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Ordre 1 théorique')

    ax2.set_xlabel('Intervalles de maillage', fontsize=12)
    ax2.set_ylabel('Ordre de convergence', fontsize=12)
    ax2.set_title('Ordres de convergence calculés', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{Nx_list[i]}→{Nx_list[i + 1]}' for i in range(len(ordres_h))])
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.legend(loc='best')
    ax2.set_ylim([0, 2])

    for i, (oh, ou) in enumerate(zip(ordres_h, ordres_u)):
        ax2.text(i - largeur / 2, oh + 0.05, f'{oh:.2f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + largeur / 2, ou + 0.05, f'{ou:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Étude de convergence - Schéma HLL (CFL={CFL}, t={T_final}s)', fontsize=16)
    plt.tight_layout()
    plt.savefig('convergence_HLL.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ÉTUDE DE CONVERGENCE DU SCHÉMA HLL POUR SAINT-VENANT")
    print("=" * 70)

    Nx_list, dx_list, erreurs_h, erreurs_u, ordres_h, ordres_u = etude_convergence()

    plot_convergence(Nx_list, dx_list, erreurs_h, erreurs_u, ordres_h, ordres_u)

    print("\n" + "=" * 70)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 70)
    print(f"{'Nx':<8} {'dx':<12} {'Erreur h':<20} {'Erreur u':<20}")
    print("-" * 70)
    for i, Nx in enumerate(Nx_list):
        print(f"{Nx:<8} {dx_list[i]:<12.6f} {erreurs_h[i]:<20.6e} {erreurs_u[i]:<20.6e}")
    print("=" * 70)