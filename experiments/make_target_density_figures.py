"""Παράγει 2D γραφήματα πυκνότητας για τους 5 προτεινόμενους στόχους.

Κάθε γράφημα δείχνει το επίπεδο των δύο πιο χαρακτηριστικών συντεταγμένων του στόχου,
με τις ακριβείς παραμέτρους που αναφέρει το thesis/proposed_targets_for_approval.md.
Τα αρχεία αποθηκεύονται στο experiments/figures/extension/target_densities/ ως PNG.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "extension", "target_densities")
os.makedirs(OUT, exist_ok=True)

CMAP = "viridis"


def density_grid(neglogp, xlim, ylim, n=400):
    """Επιστρέφει X, Y, Z=π (κανονικοποιημένο σε max=1) από U(x,y)=neglogp."""
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    X, Y = np.meshgrid(xs, ys)
    U = neglogp(X, Y)
    Z = np.exp(-(U - U.min()))
    return X, Y, Z


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# 1. Ill-conditioned Gaussian: δύο ακραίες ιδιοτιμές (λ=1 vs λ=κ).
# ΣΗΜΑΝΤΙΚΟ: ίδια κλίμακα στους δύο άξονες (set_aspect equal), αλλιώς η
# ανισοτροπία κρύβεται και η κατανομή φαίνεται ψευδώς ισοτροπική.
def fig_gaussian(kappa=100.0):
    U = lambda x, y: 0.5 * (x**2 / 1.0 + y**2 / kappa)
    lim = 3 * np.sqrt(kappa)
    X, Y, Z = density_grid(U, (-lim, lim), (-lim, lim))
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.contourf(X, Y, Z, levels=25, cmap=CMAP)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_{\min}$  ($\lambda=1$)")
    ax.set_ylabel(r"$x_{\max}$  ($\lambda=\kappa$)")
    ax.set_title(f"Ill-conditioned Gaussian ($\\kappa={int(kappa)}$)")
    save(fig, "1_gaussian.png")


# 2. Banana (twisted Gaussian) ένα 2D μπλοκ.
def fig_banana(V0=100.0, b=0.03):
    def U(x, y):
        u = y - b * x**2 + b * V0
        return 0.5 * (x**2 / V0 + u**2)
    X, Y, Z = density_grid(U, (-35, 35), (-12, 40))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.contourf(X, Y, Z, levels=25, cmap=CMAP)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_{2k}$")
    ax.set_ylabel(r"$x_{2k+1}$")
    ax.set_title(f"Banana ($V_0={int(V0)}$, $b={b}$)")
    save(fig, "2_banana.png")


# 3. Μείγμα τριών Γκαουσιανών (επίπεδο x0 - x1).
def fig_mixture(a=4.0):
    centers = [(-a, 0.0), (0.0, 0.0), (a, 0.0)]
    def U(x, y):
        comps = [np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2)) for cx, cy in centers]
        return -np.log(sum(comps))
    X, Y, Z = density_grid(U, (-a - 4, a + 4), (-4, 4))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.contourf(X, Y, Z, levels=25, cmap=CMAP)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    ax.set_title(f"Μείγμα 3 Γκαουσιανών ($a={int(a)}$)")
    save(fig, "3_mixture.png")


# 4. Neal's funnel (επίπεδο v - x1).
def fig_funnel(sigma_v=3.0):
    def U(v, x):
        return v**2 / (2 * sigma_v**2) + 0.5 * v + 0.5 * np.exp(-v) * x**2
    X, Y, Z = density_grid(U, (-9, 9), (-18, 18))
    # εδώ X=v, Y=x1
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.contourf(X, Y, Z, levels=25, cmap=CMAP)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$v$  (λαιμός)")
    ax.set_ylabel(r"$x_1\mid v$")
    ax.set_title(f"Neal's funnel ($\\sigma_v={int(sigma_v)}$)")
    save(fig, "4_funnel.png")


# 5. Double well (2D: δύο ανεξάρτητες συντεταγμένες -> 4 modes).
def fig_double_well():
    U = lambda x, y: (x**2 - np.abs(x)) + (y**2 - np.abs(y))
    X, Y, Z = density_grid(U, (-1.6, 1.6), (-1.6, 1.6))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.contourf(X, Y, Z, levels=25, cmap=CMAP)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"Double well $x^2-|x|$  ($2^2=4$ modes)")
    save(fig, "5_double_well.png")


if __name__ == "__main__":
    fig_gaussian()
    fig_banana()
    fig_mixture()
    fig_funnel()
    fig_double_well()