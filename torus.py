"""Episode - two ways to build a torus, and why they give the same space.

1. A product. A point of the circle S¹ = {z in C : |z| = 1} is an angle; a pair of points
   (z, w), one on each of two circles, is a point of S¹ × S¹. Letting z and w go around
   sweeps out a doughnut: the torus.

2. A quotient. Take the square [0,1]² and declare (0, t) ~ (1, t) and (s, 0) ~ (s, 1):
   the left edge is the same as the right edge, the bottom the same as the top. Gluing
   left to right rolls the square into a tube; gluing the two end circles bends the tube
   into a doughnut. This space is [0,1]²/~.

3. The map between them:
                      f(s, t) = (e^{2πis}, e^{2πit}).
   f is continuous and onto, and f(p) = f(q) exactly when p ~ q (an angle of 0 and of 2π
   is the same point of the circle). So f passes to the quotient as a continuous
   bijection  F : [0,1]²/~ -> S¹ × S¹.

4. The theorem. A continuous bijection from a compact space to a Hausdorff space is a
   homeomorphism (closed subsets of a compact space are compact, their images are compact,
   and compact subsets of a Hausdorff space are closed, so F maps closed sets to closed
   sets). [0,1]²/~ is compact (the image of a compact square) and S¹ × S¹ ⊂ C² is
   Hausdorff, so
                      [0,1]²/~  ≅  S¹ × S¹:
   the two constructions give the same torus  (also = R²/Z²).

This module gives f, an embedding of S¹ × S¹ in 3D space, and the shape at every moment of
the gluing, as 3D points for (u, v) in the unit square: s1 in [0, 1] rolls u into a
circle, s2 in [0, 1] bends v into a circle. At s1 = s2 = 1 the gluing is exactly the
embedded S¹ × S¹: glued(s, t, 1, 1) = embed(f(s, t)).

Run:  python -m mathscenes.torus
"""
import numpy as np


def glued(u, v, s1, s2, R_end=0.45):
    """Points (x, y, z) of the square [0,1]² part-way through gluing.
    s1: 0 flat -> 1 rolled into a tube (left edge meets right edge);
    s2: 0 straight tube -> 1 closed ring (top edge meets bottom edge)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    # roll: arc of angle 2π·s1 with the square's width kept (circumference 1)
    if s1 < 1e-4:
        a, b = u - 0.5, np.zeros_like(u)
    else:
        th = 2 * np.pi * s1 * (u - 0.5)
        r = 1 / (2 * np.pi * s1)
        a, b = r * np.sin(th), r * (1 - np.cos(th))
    # bend: center line along v becomes an arc of angle 2π·s2
    if s2 < 1e-4:
        x, y, z = a, v - 0.5, b
    else:
        ph = 2 * np.pi * s2 * (v - 0.5)
        R = (1 / (2 * np.pi * s2)) * (1 - s2) + R_end * s2 if s2 < 1 else R_end
        R = max(R, R_end)
        x = a
        y = (R - b) * np.sin(ph)
        z = R - (R - b) * np.cos(ph)
    return np.stack([x, y, z], -1)


def f(s, t):
    """The square to the product: (s, t) -> (e^{2πis}, e^{2πit})."""
    return np.exp(2j * np.pi * np.asarray(s, float)), np.exp(2j * np.pi * np.asarray(t, float))


def embed(z, w, R=0.45):
    """S¹ × S¹ -> R³: z turns around the tube, w around the ring (same shape as the fully
    glued square)."""
    s = np.angle(z) / (2 * np.pi) % 1.0
    t = np.angle(w) / (2 * np.pi) % 1.0
    return glued(s, t, 1.0, 1.0, R_end=R)


def glued_partner(s, t, tol=1e-9):
    """The other points of [0,1]² glued to (s, t) (on an edge or at a corner)."""
    ss = [s] + ([1 - s] if min(s, 1 - s) < tol else [])
    ts = [t] + ([1 - t] if min(t, 1 - t) < tol else [])
    return [(a, b) for a in ss for b in ts if (a, b) != (s, t)]


def seam_distance(p, q):
    """Largest distance between matching points of two edges (0 = glued)."""
    return float(np.max(np.linalg.norm(p - q, axis=-1)))


if __name__ == "__main__":
    t = np.linspace(0, 1, 50)
    for s1, s2 in ((0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1)):
        lr = seam_distance(glued(0 * t, t, s1, s2), glued(0 * t + 1, t, s1, s2))
        tb = seam_distance(glued(t, 0 * t, s1, s2), glued(t, 0 * t + 1, s1, s2))
        print(f"roll {s1:.1f}, bend {s2:.1f}:  left-right gap {lr:.3f}   top-bottom gap {tb:.3f}")

    # f identifies exactly the glued points
    g = np.linspace(0, 1, 41)
    S, T = np.meshgrid(g, g, indexing="ij")
    z, w = f(S, T)
    pts = np.stack([z.ravel(), w.ravel()], -1)
    same = (np.abs(pts[:, None, 0] - pts[None, :, 0]) < 1e-9) & (np.abs(pts[:, None, 1] - pts[None, :, 1]) < 1e-9)
    sv, tv = S.ravel(), T.ravel()
    ds, dt = np.abs(sv[:, None] - sv[None, :]), np.abs(tv[:, None] - tv[None, :])
    glued_rel = ((ds < 1e-9) | (np.abs(ds - 1) < 1e-9)) & ((dt < 1e-9) | (np.abs(dt - 1) < 1e-9))
    print("f(p) = f(q) exactly when p ~ q (on a 41x41 grid):", bool(np.all(same == glued_rel)))
    print("corner (0,0) is glued to", glued_partner(0.0, 0.0), " all map to", f(0, 0), f(1, 1))
    # onto: every pair of angles is hit
    a, b = np.random.default_rng(1).uniform(-np.pi, np.pi, (2, 1000))
    s0, t0 = (a / (2 * np.pi)) % 1, (b / (2 * np.pi)) % 1
    z0, w0 = f(s0, t0)
    print("onto (1000 random pairs hit):", np.allclose(z0, np.exp(1j * a)) and np.allclose(w0, np.exp(1j * b)))
    # the fully glued square is the embedded product
    u = np.random.default_rng(2).uniform(0, 1, (2, 500))
    print("glued(s,t,1,1) == embed(f(s,t)):", np.allclose(glued(u[0], u[1], 1, 1), embed(*f(u[0], u[1]))))
    # embed is injective on S¹ × S¹ (so the doughnut is an honest picture of the product)
    q = embed(*f(S[:-1, :-1].ravel(), T[:-1, :-1].ravel()))
    dmin = np.min(np.linalg.norm(q[:, None] - q[None, :], axis=-1) + np.eye(len(q)) * 9)
    print(f"distinct pairs land on distinct points (min gap {dmin:.4f} > 0):", dmin > 1e-6)
