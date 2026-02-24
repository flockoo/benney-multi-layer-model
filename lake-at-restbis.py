#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

g = 9.81

L = 100.0
Nx = 200
dx = L / Nx
CFL = 0.5
T_final = 10.0

h_lake = 2.0
zb_amplitude = 0.5


def init_conditions_lac_au_repos():
    x = (np.arange(Nx) + 0.5) * dx

    zb = zb_amplitude * np.sin(2 * np.pi * x / L)

    eta = h_lake
    h = eta - zb
    h = np.maximum(h, 0.0)

    u = np.zeros(Nx)
    q = h * u

    U = np.vstack((h, q))

    return x, zb, U


def flux_with_topography(U, zb):
    h = U[0, :]
    q = U[1, :]

    F = np.zeros_like(U)
    mask = h > 1e-8

    F[0, :] = q

    F[1, mask] = q[mask] ** 2 / h[mask] + 0.5 * g * h[mask] ** 2
    F[1, ~mask] = 0.0

    return F


def source_term(U, zb, dx):
    h = U[0, :]

    dzb_dx = np.zeros(Nx)
    dzb_dx[1:-1] = (zb[2:] - zb[:-2]) / (2 * dx)

    dzb_dx[0] = (zb[1] - zb[0]) / dx
    dzb_dx[-1] = (zb[-1] - zb[-2]) / dx

    S = np.zeros_like(U)
    S[1, :] = -g * h * dzb_dx

    return S


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


def apply_boundary_lac(U, zb):
    U[0, 0] = U[0, 1]
    U[1, 0] = -U[1, 1]

    U[0, -1] = U[0, -2]
    U[1, -1] = -U[1, -2]

    return U


def step_with_source(U, zb, dt):
    U = apply_boundary_lac(U, zb)

    F_num = np.zeros((2, Nx + 1))

    for i in range(Nx + 1):
        if i == 0:
            UL = U[:, 0]
            UR = np.array([UL[0], -UL[1]])
        elif i == Nx:
            UL = U[:, -1]
            UR = np.array([UL[0], -UL[1]])
        else:
            UL = U[:, i - 1]
            UR = U[:, i]

        F_num[:, i] = hll_flux(UL, UR)

    S = source_term(U, zb, dx)

    Unew = U.copy()
    for i in range(Nx):
        Unew[:, i] -= dt / dx * (F_num[:, i + 1] - F_num[:, i])
        Unew[:, i] += dt * S[:, i]

    Unew[0, :] = np.maximum(Unew[0, :], 0.0)
    mask = Unew[0, :] < 1e-8
    Unew[1, mask] = 0.0

    return Unew


def compute_errors(U_initial, U_final, zb):
    h_initial = U_initial[0, :]
    h_final = U_final[0, :]

    error_h = np.abs(h_final - h_initial)

    errors = {
        'Linf': np.max(error_h),
        'L1': np.mean(error_h),
        'L2': np.sqrt(np.mean(error_h ** 2)),
    }

    mass_initial = np.sum(h_initial) * dx
    mass_final = np.sum(h_final) * dx
    errors['mass_error'] = abs(mass_final - mass_initial) / mass_initial * 100

    h = U_final[0, :]
    q = U_final[1, :]
    u = np.zeros_like(h)
    mask = h > 1e-8
    u[mask] = q[mask] / h[mask]
    kinetic_energy = 0.5 * np.sum(h * u ** 2) * dx
    errors['kinetic_energy'] = kinetic_energy

    return errors


def run_simulation_lac_au_repos():
    print("=" * 60)
    print("SIMULATION LAC AU REPOS")
    print("=" * 60)
    print(f"Hauteur du lac: {h_lake} m")
    print(f"Longueur domaine: {L} m")
    print(f"Nombre de mailles: {Nx}")
    print(f"Temps final: {T_final} s")

    x, zb, U = init_conditions_lac_au_repos()
    U_initial = U.copy()
    h_initial = U[0, :].copy()

    eta_initial = h_initial + zb

    time_history = [0.0]
    error_history = [0.0]
    mass_history = [np.sum(h_initial) * dx]

    t = 0.0
    iteration = 0

    print("\nDébut de la simulation...")
    print("-" * 40)

    while t < T_final:
        dt = compute_dt(U)
        if t + dt > T_final:
            dt = T_final - t

        U = step_with_source(U, zb, dt)

        t += dt
        iteration += 1

        errors = compute_errors(U_initial, U, zb)

        time_history.append(t)
        error_history.append(errors['L2'])
        mass_history.append(np.sum(U[0, :]) * dx)

        if iteration % 100 == 0:
            print(f"Iter {iteration:5d}, t = {t:6.2f}s, dt = {dt:.2e}, "
                  f"Err L2 = {errors['L2']:.2e}, "
                  f"Mass err = {errors['mass_error']:.2e}%")

    h_final = U[0, :].copy()
    eta_final = h_final + zb
    errors_final = compute_errors(U_initial, U, zb)

    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(x, zb, 'k-', linewidth=2, label='Fond (zb)')
    ax1.plot(x, eta_initial, 'b--', linewidth=2, alpha=0.7, label='Surface initiale')
    ax1.plot(x, eta_final, 'r-', linewidth=2, label='Surface finale')
    ax1.fill_between(x, zb, eta_initial, alpha=0.1, color='blue')
    ax1.fill_between(x, zb, eta_final, alpha=0.1, color='red')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('Élévation [m]')
    ax1.set_title('Profil du lac au repos - État initial et final')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([np.min(zb) - 0.1, np.max(eta_initial) + 0.5])
    plt.show()

    return x, zb, h_initial, h_final, errors_final


def test_lac_plat():
    print("\n" + "=" * 60)
    print("TEST LAC AU REPOS SUR FOND PLAT")
    print("=" * 60)

    global zb_amplitude
    zb_amplitude = 0.0

    x, zb, U = init_conditions_lac_au_repos()
    U_initial = U.copy()
    h_initial = U[0, :].copy()

    t = 0.0
    iteration = 0

    while t < T_final / 2:
        dt = compute_dt(U)
        if t + dt > T_final / 2:
            dt = T_final / 2 - t

        U = step_with_source(U, zb, dt)
        t += dt
        iteration += 1

    errors = compute_errors(U_initial, U, zb)

    print(f"\nRésultats fond plat:")
    print(f"Erreur L2: {errors['L2']:.2e}")
    print(f"Erreur masse: {errors['mass_error']:.2e}%")




if __name__ == "__main__":
    x, zb, h_initial, h_final, errors = run_simulation_lac_au_repos()