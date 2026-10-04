"""Episode - Newton's method and its three basins.

To solve z³ = 1, start anywhere in the complex plane and repeat Newton's step

    z  ->  z − (z³ − 1) / (3z²)

Almost every start runs to one of the three cube roots of 1. Color each start by the
root it reaches and you get three basins of attraction. Their common boundary is a
fractal (a Julia set of the Newton map) with a strange property: every point on it
touches all three basins. Wherever two colors meet, the third is there too, at every
scale ("Wada" property of the boundary).

A boundary point you can write down: −2^(−1/3) ≈ −0.7937 is sent by one step to 0, where
the method breaks (z² = 0), so it belongs to no basin. Zoom in around it and all three
colors keep appearing.

Run:  python -m mathscenes.newton
"""
import numpy as np

ROOTS = np.exp(2j * np.pi * np.arange(3) / 3)          # 1, e^{2πi/3}, e^{4πi/3}


def newton_step(z):
    with np.errstate(all="ignore"):
        return z - (z ** 3 - 1) / (3 * z ** 2)


def iterate(z0, steps):
    """All iterates: array (steps + 1, *z0.shape)."""
    out = [np.asarray(z0, complex)]
    for _ in range(steps):
        out.append(newton_step(out[-1]))
    return np.array(out)


def nearest_root(z):
    """Index (0, 1, 2) of the closest cube root of 1, and the distance to it."""
    d = np.abs(np.asarray(z)[..., None] - ROOTS)
    d = np.nan_to_num(d, nan=9.0, posinf=9.0)
    return d.argmin(-1), d.min(-1)


def basins(z0, steps=40, tol=1e-6):
    """Root index each start converges to (−1 if not within tol), and how many steps it took."""
    z = np.asarray(z0, complex).copy()
    k = -np.ones(z.shape, int)
    when = np.full(z.shape, steps, int)
    for i in range(steps):
        z = newton_step(z)
        idx, d = nearest_root(z)
        new = (d < tol) & (k < 0)
        k[new], when[new] = idx[new], i + 1
    return k, when


if __name__ == "__main__":
    z = np.linspace(-1.5, 1.5, 601)
    Z = z[None, :] + 1j * z[:, None]
    k, when = basins(Z)
    print("fraction of starts in each basin:", [round(float((k == j).mean()), 3) for j in range(3)],
          " unresolved:", round(float((k < 0).mean()), 4))
    b = -2 ** (-1 / 3)
    print(f"one step from {b:.4f}: {newton_step(b + 0j):.2e}  (lands on 0, where the method breaks)")
    for r in (1e-1, 1e-2, 1e-3, 1e-4):            # all three basins near that point, at every scale
        th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        ring = b + r * np.exp(1j * th) * np.linspace(0.05, 1, 40)[:, None]
        kk, _ = basins(ring.ravel(), steps=80)
        print(f"  within {r:g} of it: basins present {sorted(set(kk[kk >= 0].tolist()))}")
