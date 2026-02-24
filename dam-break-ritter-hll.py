import numpy as np
import matplotlib.pyplot as plt

g = 9.81

L = 1.0
Nx = 400
dx = L / Nx
CFL = 0.5
T_final = 0.2

h0 = 1.0
x_bar = 0.5


def init_conditions():
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


def apply_boundary(U):
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]
    return U


def step(U, dt):
    U = apply_boundary(U)

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


def run_simulation():
    x, U = init_conditions()
    h_initial = U[0, :].copy()

    t = 0.0
    iteration = 0

    print("Début de la simulation...")
    print("=" * 50)

    while t < T_final:
        dt = compute_dt(U)
        if t + dt > T_final:
            dt = T_final - t
        U = step(U, dt)
        t += dt
        iteration += 1

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, t = {t:.4f}s, dt = {dt:.2e}")

    h_final = U[0, :].copy()

    h_analytique, u_analytique = solution_analytique_ritter(x, T_final, h0, x_bar)

    L2_error_h = compute_L2_error(h_final, h_analytique, dx)

    h = U[0, :]
    u_numeric = np.zeros_like(h)
    mask = h > 1e-8
    u_numeric[mask] = U[1, mask] / h[mask]
    L2_error_u = compute_L2_error(u_numeric, u_analytique, dx)

    print("\n" + "=" * 50)
    print("RÉSULTATS FINAUX")
    print("=" * 50)
    print(f"Temps final : {T_final} s")
    print(f"Nombre d'itérations : {iteration}")
    print(f"Nombre de mailles : {Nx}")
    print(f"dx = {dx:.4f} m")
    print(f"CFL = {CFL}")
    print("\n--- ERREURS L2 ---")
    print(f"Erreur L2 sur h : {L2_error_h:.6e}")
    print(f"Erreur L2 sur u : {L2_error_u:.6e}")

    print("\n--- STATISTIQUES ---")
    print(f"Hauteur max numérique : {np.max(h_final):.4f} m")
    print(f"Hauteur max analytique : {np.max(h_analytique):.4f} m")
    print(f"Position front onde numérique : {x[np.where(h_final > 1e-3)[0][-1]]:.3f} m")
    print(f"Position front onde théorique : {x_bar + 2 * np.sqrt(g * h0) * T_final:.3f} m")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x, h_initial, 'gray', linestyle='--', label="Initiale (t=0)", alpha=0.7)
    axes[0, 0].plot(x, h_final, 'b-', linewidth=2, label="Numérique")
    axes[0, 0].plot(x, h_analytique, 'r--', linewidth=1.5, label="Analytique", alpha=0.8)
    axes[0, 0].axvline(x=x_bar, color='k', linestyle=':', label="Barrage")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("h(x,t) [m]")
    axes[0, 0].set_title(f"Hauteur d'eau à t = {T_final} s")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    error_abs = np.abs(h_final - h_analytique)
    axes[0, 1].plot(x, error_abs, 'r-', linewidth=1.5)
    axes[0, 1].fill_between(x, 0, error_abs, alpha=0.3, color='red')
    axes[0, 1].set_xlabel("x [m]")
    axes[0, 1].set_ylabel("|h_num - h_ana| [m]")
    axes[0, 1].set_title("Erreur absolue")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.1 * np.max(error_abs)])

    axes[1, 0].plot(x, u_numeric, 'b-', linewidth=2, label="Numérique")
    axes[1, 0].plot(x, u_analytique, 'r--', linewidth=1.5, label="Analytique", alpha=0.8)
    axes[1, 0].axvline(x=x_bar, color='k', linestyle=':')
    axes[1, 0].set_xlabel("x [m]")
    axes[1, 0].set_ylabel("u(x,t) [m/s]")
    axes[1, 0].set_title(f"Vitesse à t = {T_final} s")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    q_numeric = h_final * u_numeric
    q_analytique = h_analytique * u_analytique
    axes[1, 1].plot(x, q_numeric, 'b-', linewidth=2, label="Numérique")
    axes[1, 1].plot(x, q_analytique, 'r--', linewidth=1.5, label="Analytique", alpha=0.8)
    axes[1, 1].axvline(x=x_bar, color='k', linestyle=':')
    axes[1, 1].set_xlabel("x [m]")
    axes[1, 1].set_ylabel("q(x,t) [m²/s]")
    axes[1, 1].set_title("Débit h × u")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Rupture de barrage - Schéma HLL explicite (t={T_final}s, Nx={Nx}, CFL={CFL})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return L2_error_h, L2_error_u


def convergence_study():
    resolutions = [50, 100, 200, 400, 800]
    L2_errors_h = []
    L2_errors_u = []
    dx_values = []

    print("Étude de convergence L2")
    print("=" * 50)

    global Nx, dx

    for nx in resolutions:
        print(f"\nRésolution : Nx = {nx}")

        Nx = nx
        dx = L / Nx

        L2_h, L2_u = run_simulation()

        L2_errors_h.append(L2_h)
        L2_errors_u.append(L2_u)
        dx_values.append(dx)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.loglog(dx_values, L2_errors_h, 'bo-', linewidth=2, markersize=8, label='Erreur L2(h)')

    order1 = [L2_errors_h[0] * (dx / dx_values[0]) for dx in dx_values]
    order2 = [L2_errors_h[0] * (dx / dx_values[0]) ** 2 for dx in dx_values]

    plt.loglog(dx_values, order1, 'k--', label='Ordre 1', alpha=0.5)
    plt.loglog(dx_values, order2, 'k:', label='Ordre 2', alpha=0.5)

    plt.xlabel('dx [m]')
    plt.ylabel('Erreur L2')
    plt.title('Convergence de l\'erreur sur h')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    plt.subplot(1, 2, 2)
    plt.loglog(dx_values, L2_errors_u, 'ro-', linewidth=2, markersize=8, label='Erreur L2(u)')
    plt.loglog(dx_values, order1, 'k--', label='Ordre 1', alpha=0.5)
    plt.loglog(dx_values, order2, 'k:', label='Ordre 2', alpha=0.5)

    plt.xlabel('dx [m]')
    plt.ylabel('Erreur L2')
    plt.title('Convergence de l\'erreur sur u')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    L2_h, L2_u = run_simulation()