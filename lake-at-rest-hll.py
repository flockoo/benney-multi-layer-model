import numpy as np
import matplotlib.pyplot as plt

g = 9.81
Lx = 10.0
Nx = 200
dx = Lx / Nx
x = np.linspace(dx/2, Lx-dx/2, Nx)

b = 0.2 * np.exp(- (x-5)**2 / 1.0)

C = 1.0
h = C - b
h = np.maximum(h, 1e-6)
q = np.zeros_like(h)

def apply_bc(h, q):
    h_ext = np.zeros(len(h) + 2)
    q_ext = np.zeros(len(q) + 2)

    h_ext[1:-1] = h
    q_ext[1:-1] = q

    h_ext[0] = h[0]
    h_ext[-1] = h[-1]
    q_ext[0] = 0.0
    q_ext[-1] = 0.0

    return h_ext, q_ext

def primitive_flux(h, q):
    F1 = q
    F2 = np.zeros_like(h)
    mask = h > 1e-12
    F2[mask] = q[mask]**2 / h[mask] + 0.5 * g * h[mask]**2
    return F1, F2

def wave_speeds(h_L, q_L, h_R, q_R):
    u_L = q_L / h_L if h_L > 1e-12 else 0.0
    u_R = q_R / h_R if h_R > 1e-12 else 0.0

    c_L = np.sqrt(g * h_L) if h_L > 1e-12 else 0.0
    c_R = np.sqrt(g * h_R) if h_R > 1e-12 else 0.0

    S_L = min(u_L - c_L, u_R - c_R)
    S_R = max(u_L + c_L, u_R + c_R)

    return S_L, S_R

def hll_flux(h_L, q_L, h_R, q_R):
    U_L = np.array([h_L, q_L])
    U_R = np.array([h_R, q_R])

    F1_L, F2_L = primitive_flux(h_L, q_L)
    F1_R, F2_R = primitive_flux(h_R, q_R)
    F_L = np.array([F1_L, F2_L])
    F_R = np.array([F1_R, F2_R])

    S_L, S_R = wave_speeds(h_L, q_L, h_R, q_R)

    if S_L >= 0:
        return F_L
    elif S_R <= 0:
        return F_R
    else:
        F_HLL = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L + 1e-14)
        return F_HLL

def compute_dt(h, q, dx, CFL):
    u = np.zeros_like(h)
    mask = h > 1e-12
    u[mask] = q[mask] / h[mask]
    c = np.sqrt(g * h)
    max_speed = np.max(np.abs(u) + c)
    return CFL * dx / (max_speed + 1e-14)

def step_hll(h, q, b, dt, dx):
    h_ext, q_ext = apply_bc(h, q)
    b_ext, _ = apply_bc(b, b)

    h_new = np.zeros_like(h)
    q_new = np.zeros_like(q)

    for i in range(1, len(h_ext)-1):
        h_L = h_ext[i]
        q_L = q_ext[i]
        h_R = h_ext[i+1]
        q_R = q_ext[i+1]
        b_L = b_ext[i]
        b_R = b_ext[i+1]

        F_hll = hll_flux(h_L, q_L, h_R, q_R)

        if i > 1:
            h_L_prev = h_ext[i-1]
            q_L_prev = q_ext[i-1]
            h_R_prev = h_ext[i]
            q_R_prev = q_ext[i]
            F_hll_prev = hll_flux(h_L_prev, q_L_prev, h_R_prev, q_R_prev)
        else:
            F_hll_prev = np.zeros(2)

        h_new[i-1] = h_ext[i] - (dt/dx) * (F_hll[0] - F_hll_prev[0])
        q_new[i-1] = q_ext[i] - (dt/dx) * (F_hll[1] - F_hll_prev[1])

        if i > 1 and i < len(h_ext)-1:
            dbdx = (b_ext[i+1] - b_ext[i-1]) / (2*dx)
            h_avg = 0.5 * (h_ext[i-1] + h_ext[i+1])
            q_new[i-1] += dt * (-g * h_avg * dbdx)

    q_new[0] = 0.0
    q_new[-1] = 0.0
    h_new = np.maximum(h_new, 1e-8)

    return h_new, q_new

t = 0.0
t_end = 2.0
CFL = 0.5

h_history = [h.copy()]
t_history = [t]

print("Début de la simulation avec schéma HLL...")
while t < t_end:
    dt = compute_dt(h, q, dx, CFL)
    if t + dt > t_end:
        dt = t_end - t

    h, q = step_hll(h, q, b, dt, dx)

    t += dt

    if len(h_history) < 10 or t_history[-1] + 0.2 < t:
        h_history.append(h.copy())
        t_history.append(t)

    if int(t * 10) % 2 == 0 and t > t_history[-2] + 0.1:
        print(f"t = {t:.3f}, dt = {dt:.6f}, min(h) = {np.min(h):.6f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, np.ones_like(x) * C, 'k--', label='Surface initiale η = C', linewidth=2)
plt.plot(x, h + b, 'r-', label='Surface finale η(x,t)', linewidth=1.5)
plt.plot(x, b, 'b-', label='Fond b(x)', linewidth=1.5)
plt.fill_between(x, b, h+b, color='lightblue', alpha=0.5, label='Eau')
plt.xlabel('x')
plt.ylabel('Hauteur')
plt.title('État final - Schéma HLL')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
for i, h_val in enumerate(h_history):
    alpha = 0.3 + 0.7 * i / len(h_history)
    plt.plot(x, h_val + b, alpha=alpha, label=f't={t_history[i]:.2f}' if i%2==0 else "")
plt.plot(x, b, 'k-', linewidth=2, label='Fond')
plt.xlabel('x')
plt.ylabel('Surface libre η')
plt.title('Évolution de la surface libre')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
eta_error = [np.max(np.abs(h_val + b - C)) for h_val in h_history]
plt.plot(t_history, eta_error, 'ro-', markersize=4, linewidth=1.5)
plt.xlabel('Temps')
plt.ylabel('max|η - C|')
plt.title('Erreur sur l état au repos')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ANALYSE DES RÉSULTATS - SCHÉMA HLL")
print("="*50)

mass_initial = np.sum(h_history[0] * dx)
mass_final = np.sum(h * dx)
mass_error = abs(mass_final - mass_initial) / mass_initial
print(f"Masse initiale: {mass_initial:.8f}")
print(f"Masse finale:   {mass_final:.8f}")
print(f"Erreur relative: {mass_error:.2e}")

final_error = np.max(np.abs(h + b - C))
print(f"\nErreur max sur surface libre: {final_error:.2e}")

if final_error < 1e-4:
    print("✓ Excellent - schéma quasi well-balanced")
elif final_error < 1e-3:
    print("✓ Bon - erreur acceptable")
elif final_error < 1e-2:
    print("∼ Moyen - des améliorations sont possibles")
else:
    print("✗ À améliorer - schéma pas assez well-balanced")

eta0 = C * np.ones_like(x)
h0 = C - b

plt.figure(figsize=(8,4))
plt.plot(x, eta0, 'k--', label='Surface initiale η = C (plate)')
plt.plot(x, h0, 'r', label='h(x,0) = C - b (creux au centre)')
plt.plot(x, b, 'b', label='Fond b(x) (bosse au centre)')
plt.legend()
plt.xlabel('x')
plt.ylabel('hauteur')
plt.title('Vérification : η plate, h creux à cause de b')
plt.grid(True)
plt.show()