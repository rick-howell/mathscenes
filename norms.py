"""Episode - the parallelogram law.

In a parallelogram with sides x and y, the diagonals are x + y and x − y. With ordinary
lengths, the squares of the two diagonals add up to the squares of all four sides:

    ‖x + y‖² + ‖x − y‖² = 2‖x‖² + 2‖y‖²            (the parallelogram law)

A remarkable fact (Jordan and von Neumann, 1935): if a norm satisfies this law for all
x and y, then it comes in the usual way from an inner product, ‖x‖ = √⟨x, x⟩, and the
inner product can be read off from lengths alone (polarisation):

    ⟨x, y⟩ = (‖x + y‖² − ‖x − y‖²) / 4

The p-norms on ℝ²,

    ‖x‖_p = (|x₁|^p + |x₂|^p)^(1/p)        ‖x‖_∞ = max(|x₁|, |x₂|),

have unit balls from a diamond (p = 1) through the circle (p = 2) to a square (p = ∞).
The law holds for the p-norm if and only if p = 2, the Euclidean norm. One pair is
enough to see it fail: for x = (1, 0), y = (0, 1) the two sides are 2·4^(1/p) and 4,
equal only when p = 2.

Run:  python -m mathscenes.norms
"""
import numpy as np

INF = np.inf


def pnorm(v, p):
    """p-norm of 2-vectors; v has shape (..., 2). p may be np.inf."""
    a = np.abs(np.asarray(v, float))
    if np.isinf(p):
        return a.max(-1)
    m = a.max(-1)
    safe = np.where(m > 0, m, 1.0)
    return np.where(m > 0, safe * ((a / safe[..., None]) ** p).sum(-1) ** (1.0 / p), 0.0)


def p_from_q(q):
    """Animations run on q = 1/p in [0, 1]: q = 1 is p = 1, q = 1/2 is p = 2, q = 0 is p = ∞."""
    return INF if q <= 1e-6 else 1.0 / q


def unit_ball(p, n=720):
    """Points on the boundary of the unit p-ball, as complex numbers."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    d = np.stack([np.cos(th), np.sin(th)], -1)
    return (1.0 / pnorm(d, p)) * (d[:, 0] + 1j * d[:, 1])


def parallelogram_sides(x, y, p):
    """(‖x+y‖² + ‖x−y‖², 2‖x‖² + 2‖y‖²) for the p-norm: the two diagonals vs the four sides."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    lhs = pnorm(x + y, p) ** 2 + pnorm(x - y, p) ** 2
    rhs = 2 * pnorm(x, p) ** 2 + 2 * pnorm(y, p) ** 2
    return float(lhs), float(rhs)


def polarisation(x, y, p):
    """(‖x+y‖² − ‖x−y‖²)/4: the inner product, when the norm has one."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float((pnorm(x + y, p) ** 2 - pnorm(x - y, p) ** 2) / 4)


def worst_violation(p, pairs=2000, seed=0):
    """Largest relative failure of the law over random pairs (0 means it holds)."""
    rng = np.random.default_rng(seed)
    X, Y = rng.normal(size=(pairs, 2)), rng.normal(size=(pairs, 2))
    lhs = pnorm(X + Y, p) ** 2 + pnorm(X - Y, p) ** 2
    rhs = 2 * pnorm(X, p) ** 2 + 2 * pnorm(Y, p) ** 2
    return float(np.max(np.abs(lhs - rhs) / rhs))


if __name__ == "__main__":
    print("x = (1, 0), y = (0, 1):   diagonals²  vs  sides²")
    for p in (1, 1.5, 2, 3, INF):
        lhs, rhs = parallelogram_sides((1, 0), (0, 1), p)
        print(f"  p = {p:>4}:   {lhs:.4f}  vs  {rhs:.4f}")
    print("largest relative violation over 2000 random pairs:")
    for p in (1, 1.5, 1.9, 2, 2.1, 3, INF):
        print(f"  p = {p:>4}:  {worst_violation(p):.2e}")
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(1000, 2)), rng.normal(size=(1000, 2))
    err = max(abs(polarisation(a, b, 2) - a @ b) for a, b in zip(X, Y))
    print(f"p = 2: polarisation gives back the dot product (largest error {err:.1e})")
