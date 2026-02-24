import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time

def calculate_vertical_matrices():
    B = np.array([[1/6, 1/12, 0],
                  [1/12, 1/3, 1/12],
                  [0, 1/12, 1/6]])

    D = np.array([[-0.5, 0.5, 0],
                  [-0.5, 0, 0.5],
                  [0, -0.5, 0.5]])

    E = np.array([-1.0, 0.0, 1.0])

    C_dict = {}
    vals = {
        (0,0,0): 1/24, (0,0,1): 1/24,
        (0,1,1): 1/12, (1,1,1): 1/4,
        (1,1,2): 1/12, (1,2,2): 1/12,
        (2,2,2): 1/24
    }
    for (i,k,j), val in vals.items():
        C_dict[(i,k,j)] = val
        C_dict[(i,j,k)] = val
        C_dict[(k,i,j)] = val
        C_dict[(k,j,i)] = val
        C_dict[(j,i,k)] = val
        C_dict[(j,k,i)] = val
    return B, D, E, C_dict

B, D, E, C_dict = calculate_vertical_matrices()

def hll_flux(UL, UR, g=9.81):
    hL, q0L, q1L, q2L, K1L = UL
    hR, q0R, q1R, q2R, K1R = UR

    hL_safe = max(hL, 1e-8)
    hR_safe = max(hR, 1e-8)

    u0L = q0L / hL_safe
    u1L = q1L / hL_safe
    u2L = q2L / hL_safe
    u0R = q0R / hR_safe
    u1R = q1R / hR_safe
    u2R = q2R / hR_safe

    cL = np.sqrt(g * hL_safe)
    cR = np.sqrt(g * hR_safe)

    sL = min(u0L - cL, u1L - cL, u2L - cL, u0R - cR, u1R - cR, u2R - cR)
    sR = max(u0L + cL, u1L + cL, u2L + cL, u0R + cR, u1R + cR, u2R + cR)

    FL = np.array([q0L,
                   q0L*u0L + 0.5*g*hL**2,
                   q1L*u1L + 0.5*g*hL**2,
                   q2L*u2L + 0.5*g*hL**2,
                   K1L * u1L])
    FR = np.array([q0R,
                   q0R*u0R + 0.5*g*hR**2,
                   q1R*u1R + 0.5*g*hR**2,
                   q2R*u2R + 0.5*g*hR**2,
                   K1R * u1R])

    if sL >= 0:
        return FL
    elif sR <= 0:
        return FR
    else:
        UR_array = np.array([hR, q0R, q1R, q2R, K1R])
        UL_array = np.array([hL, q0L, q1L, q2L, K1L])
        return (sR * FL - sL * FR + sL * sR * (UR_array - UL_array)) / (sR - sL)

def apply_bc_mur(U):
    U_bc = U.copy()
    U_bc[1, 0] = -U[1, 1]
    U_bc[2, 0] = -U[2, 1]
    U_bc[3, 0] = -U[3, 1]
    U_bc[0, 0] = U[0, 1]
    U_bc[4, 0] = U[4, 1]
    U_bc[:, -1] = U[:, -2]
    return U_bc

def compute_source_vectorized(U, dx, g=9.81, eps=0.1):
    Nx = U.shape[1]
    S = np.zeros_like(U)

    h = U[0, :]
    q0, q1, q2, K1 = U[1, :], U[2, :], U[3, :], U[4, :]

    mask = h > 1e-8
    h_safe = np.where(mask, h, 1e-8)

    u0 = q0 / h_safe
    u1 = q1 / h_safe
    u2 = q2 / h_safe

    kappa1 = np.where(mask, K1 / h, 0.0)

    u_kappa = np.zeros((3, Nx))
    u_kappa[0, :] = u0 * 0.0
    u_kappa[1, :] = u1 * kappa1
    u_kappa[2, :] = u2 * 0.0

    source_momentum = D @ u_kappa

    S[1, :] = source_momentum[0, :]
    S[2, :] = source_momentum[1, :]
    S[3, :] = source_momentum[2, :]

    M_11 = B[1, 1] * h
    S[4, :] = np.where(mask,
                       -(1.0/(eps * h)) * M_11 * kappa1 - (kappa1/h) * E[1] * kappa1,
                       0.0)

    return S

def simulate(U0, T_final, L, Nx, CFL, g, bc_type='mur', save_freq=0.05):
    dx = L / Nx
    x_centers = np.linspace(dx/2, L - dx/2, Nx)

    U = U0.copy()
    t = 0.0
    n_step = 0
    next_save = save_freq

    h_history = [U[0,:].copy()]
    t_history = [t]

    print(f"Début simulation avec Nx={Nx}, T_final={T_final}s")
    start_time = time.time()

    while t < T_final:
        U = apply_bc_mur(U)

        h = U[0, :]
        h_safe = np.maximum(h, 1e-8)
        u0 = U[1, :] / h_safe
        u1 = U[2, :] / h_safe
        u2 = U[3, :] / h_safe

        c = np.sqrt(g * h_safe)
        u_max_local = np.maximum(np.abs(u0), np.maximum(np.abs(u1), np.abs(u2))) + c
        u_max = np.max(u_max_local)

        dt = CFL * dx / (u_max + 1e-8)
        if t + dt > T_final:
            dt = T_final - t

        F_num = np.zeros((5, Nx+1))

        for i in range(1, Nx):
            F_num[:, i] = hll_flux(U[:, i-1], U[:, i], g)

        U_fictif = U[:, 0].copy()
        U_fictif[1:4] = -U_fictif[1:4]
        F_num[:, 0] = hll_flux(U_fictif, U[:, 0], g)

        F_num[:, Nx] = hll_flux(U[:, -1], U[:, -1], g)

        S = compute_source_vectorized(U, dx, g)

        U_new = U - (dt/dx) * (F_num[:, 1:] - F_num[:, :-1]) + dt * S

        U_new[0, :] = np.maximum(U_new[0, :], 1e-8)

        U = U_new
        t += dt
        n_step += 1

        if t >= next_save:
            h_history.append(U[0,:].copy())
            t_history.append(t)
            next_save += save_freq
            print(f"  t = {t:.3f}s, dt = {dt:.5f}s, pas {n_step}")

    elapsed = time.time() - start_time
    print(f"Simulation terminée: {n_step} pas de temps en {elapsed:.1f}s")

    return x_centers, U, t_history, h_history

def hll_flux_mono(UL, UR, g):
    hL, qL = UL
    hR, qR = UR
    uL = qL/hL if hL>1e-8 else 0
    uR = qR/hR if hR>1e-8 else 0
    cL = np.sqrt(g*hL); cR = np.sqrt(g*hR)
    sL = min(uL-cL, uR-cR); sR = max(uL+cL, uR+cR)
    FL = np.array([qL, qL*uL + 0.5*g*hL**2])
    FR = np.array([qR, qR*uR + 0.5*g*hR**2])
    if sL >= 0:
        return FL
    elif sR <= 0:
        return FR
    else:
        return (sR*FL - sL*FR + sL*sR*(np.array([hR, qR]) - np.array([hL, qL])))/(sR-sL)

def step_mono(U, dt, dx, g):
    Nx = U.shape[1]

    U_bc = U.copy()
    U_bc[1, 0] = -U[1, 1]
    U_bc[0, 0] = U[0, 1]
    U_bc[:, -1] = U[:, -2]

    F_num = np.zeros((2, Nx+1))

    U_fictif = np.array([U_bc[0,0], -U_bc[1,0]])
    F_num[:, 0] = hll_flux_mono(U_fictif, U_bc[:,0], g)

    for i in range(1, Nx):
        F_num[:, i] = hll_flux_mono(U_bc[:, i-1], U_bc[:, i], g)

    F_num[:, Nx] = hll_flux_mono(U_bc[:, -1], U_bc[:, -1], g)

    U_new = U - (dt/dx) * (F_num[:, 1:] - F_num[:, :-1])
    U_new[0, :] = np.maximum(U_new[0, :], 1e-8)

    return U_new

def simulate_mono(U0, T_final, L, Nx, CFL, g):
    dx = L / Nx
    U = U0.copy()
    t = 0

    while t < T_final:
        h = U[0, :]
        h_safe = np.maximum(h, 1e-8)
        u = U[1, :] / h_safe
        c = np.sqrt(g * h_safe)
        vmax = np.max(np.abs(u) + c)
        dt = CFL * dx / (vmax + 1e-8)

        if t + dt > T_final:
            dt = T_final - t

        U = step_mono(U, dt, dx, g)
        t += dt

    return U

def test_reduction_monocouche():
    print("="*60)
    print("Test 1 : Reduction au modele mono-couche")
    print("="*60)

    L = 2.0
    Nx = 200
    g = 9.81
    CFL = 0.5
    T_final = 0.2
    x_bar = 0.75
    h0 = 1.0

    dx = L / Nx
    x_centers = np.linspace(dx/2, L - dx/2, Nx)

    U_multi = np.zeros((5, Nx))
    U_multi[0, :] = np.where(x_centers <= x_bar, h0, 0.0)
    U_multi[1:4, :] = 0.0
    U_multi[4, :] = 0.0

    x, U_final_multi, t_h, h_hist = simulate(U_multi, T_final, L, Nx, CFL, g)

    U_mono = np.zeros((2, Nx))
    U_mono[0, :] = np.where(x_centers <= x_bar, h0, 0.0)
    U_mono[1, :] = 0.0

    U_mono = simulate_mono(U_mono, T_final, L, Nx, CFL, g)

    h_multi = U_final_multi[0, :]
    h_mono = U_mono[0, :]

    err_h = np.linalg.norm(h_multi - h_mono) / np.linalg.norm(h_mono)
    print(f"Erreur relative sur h : {err_h:.2e}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_centers, h_multi, 'b-', label='Multi-couches', linewidth=1.5)
    ax.plot(x_centers, h_mono, 'r--', label='Mono-couche', linewidth=1.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Hauteur h (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Comparaison des hauteurs (t={T_final}s)')
    plt.tight_layout()
    plt.savefig('validation_reduction.png', dpi=150)
    plt.show(block=False)
    plt.pause(0.1)

    return err_h

def test_profil_non_uniforme():
    print("\n" + "="*60)
    print("Test 2 : Profil de vitesse non-uniforme (cisaillement)")
    print("="*60)

    L = 2.0
    Nx = 100
    g = 9.81
    CFL = 0.5
    T_final = 0.1
    x_bar = 0.75
    h0 = 1.0

    dx = L / Nx
    x_centers = np.linspace(dx/2, L - dx/2, Nx)

    U_multi = np.zeros((5, Nx))
    U_multi[0, :] = np.where(x_centers <= x_bar, h0, 0.0)

    u0_init = 0.1
    u1_init = 0.3
    u2_init = 0.5
    U_multi[1, :] = U_multi[0, :] * u0_init
    U_multi[2, :] = U_multi[0, :] * u1_init
    U_multi[3, :] = U_multi[0, :] * u2_init
    U_multi[4, :] = 0.0

    x, U_final, t_h, h_hist = simulate(U_multi, T_final, L, Nx, CFL, g)

    h_final = U_final[0, :]
    h_safe = np.maximum(h_final, 1e-8)
    u0_final = U_final[1, :]/h_safe
    u1_final = U_final[2, :]/h_safe
    u2_final = U_final[3, :]/h_safe
    u_mean = 0.25*u0_final + 0.5*u1_final + 0.25*u2_final

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(x_centers, h_final, 'k-', linewidth=2, label='Hauteur h')
    axes[0].set_ylabel('Hauteur h (m)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Profil de hauteur (t={T_final}s)')

    axes[1].plot(x_centers, u0_final, 'b-', label='u_fond (ξ=0)', linewidth=1.5)
    axes[1].plot(x_centers, u1_final, 'g-', label='u_milieu (ξ=0.5)', linewidth=1.5)
    axes[1].plot(x_centers, u2_final, 'r-', label='u_surface (ξ=1)', linewidth=1.5)
    axes[1].plot(x_centers, u_mean, 'k--', label='u_moyenne', linewidth=2)
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('Vitesse u (m/s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Profils de vitesse verticaux')

    plt.suptitle("Test de profil cisaillé")
    plt.tight_layout()
    plt.savefig('validation_cisaillement.png', dpi=150)
    plt.show(block=False)
    plt.pause(0.1)

    return x_centers, U_final

if __name__ == "__main__":
    print("VALIDATION DU MODÈLE BI-COUCHE")
    print("="*60)

    plt.ion()

    try:
        err_h = test_reduction_monocouche()
        x, U_final = test_profil_non_uniforme()

        print("\n" + "="*60)
        print("RÉSUMÉ DE LA VALIDATION")
        print("="*60)
        print(f"✓ Réduction au mono-couche : {err_h:.2e}")

        print("\nAppuie sur Entrée pour fermer les figures...")
        input()

    finally:
        plt.close('all')