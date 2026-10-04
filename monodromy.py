"""Episode 1 - Monodromy.

Take the family of polynomials  p_c(z) = z^5 - z + c.  For most values of c it has
five distinct roots. The exceptions are the four values of c where two roots collide
(the zeros of the discriminant). Move c once around a closed loop that avoids those
points: every root travels along a continuous path and the set of roots comes back to
itself - but individual roots may have swapped places. The resulting permutation is
the *monodromy* of the loop. Different loops give different permutations, and
together they generate the Galois group of z^5 - z + c over C(c), which is all of S_5.

Run this file to print the permutation and (with matplotlib) save a picture:
    python -m mathscenes.monodromy
"""
import numpy as np

from .common import complex_grid, level_lines, permutation_cycles, track_continuously

DEGREE = 5


def p(z, c):
    """p_c(z) = z^5 - z + c, evaluated elementwise."""
    return z ** 5 - z + c


def branch_points():
    """Values of c where p_c has a repeated root.

    A double root z satisfies p_c(z) = 0 and p_c'(z) = 5 z^4 - 1 = 0, so
    z^4 = 1/5 and c = z - z^5 = (4/5) z.  That gives four points on a circle of
    radius (4/5) * 5^(-1/4) ~ 0.535.
    """
    z = 5 ** -0.25 * np.exp(0.5j * np.pi * np.arange(4))
    return 0.8 * z


def figure_eight(steps=4000, a=0.75, b=0.45):
    """The loop used in the video: c(s) = a cos(2 pi s) + i b sin(4 pi s), s in [0, 1].
    It winds around the branch points on the right and on the left in opposite
    senses, which is why the answer is a 3-cycle rather than something simpler."""
    s = np.linspace(0.0, 1.0, steps)
    return a * np.cos(2 * np.pi * s) + 1j * b * np.sin(4 * np.pi * s)


def roots(c):
    """The five roots of p_c (unordered)."""
    return np.roots([1, 0, 0, 0, -1, c])


def root_tracks(cpath):
    """Continuous paths of the five roots as c follows cpath. Shape (len(cpath), 5).
    Roots are labelled 1..5 counter-clockwise by angle at the start of the loop."""
    sets = np.array([roots(c) for c in cpath])
    sets[0] = sets[0][np.argsort(np.angle(sets[0]))]
    return track_continuously(sets)


def monodromy(cpath):
    """(perm, cycles) for the loop cpath (which must be closed)."""
    tr = root_tracks(cpath)
    return permutation_cycles(tr[0], tr[-1])


def field(c, width, height, span=2.9):
    """p_c on a pixel grid, plus its contour structure (see common.level_lines)."""
    Z = complex_grid(width, height, span=span)
    F = p(Z, c)
    return F, level_lines(F)


def preimages_of_circles(c, radii, spokes=26, twist=0.0):
    """All z with p_c(z) = w for w on concentric circles |w| = r (a polar grid in the
    w-plane pulled back to the z-plane). Returns (len(radii)*spokes, 5) complex."""
    ws = []
    for j, r in enumerate(radii):
        th = 2 * np.pi * (np.arange(spokes) + 0.5 * (j % 2)) / spokes + twist
        ws.append(r * np.exp(1j * th))
    ws = np.concatenate(ws)
    n = len(ws)
    M = np.zeros((n, 5, 5), complex)          # companion matrices, solved in one batch
    M[:, 1:, :4] = np.eye(4)
    M[:, 0, 4] = -(c - ws)
    M[:, 1, 4] = 1.0
    return np.linalg.eigvals(M)


if __name__ == "__main__":
    cpath = figure_eight()
    tr = root_tracks(cpath)
    perm, cycles = permutation_cycles(tr[0], tr[-1])
    print("branch points:", np.round(branch_points(), 4))
    print("permutation:", perm, " cycles:", cycles)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to also save a picture)")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(cpath.real, cpath.imag, "k-", lw=1)
        bp = branch_points()
        ax[0].plot(bp.real, bp.imag, "rx", ms=10)
        ax[0].set_title("loop in the c-plane (x = branch points)")
        for i in range(5):
            ax[1].plot(tr[:, i].real, tr[:, i].imag, lw=1.5, label=f"root {i + 1}")
            ax[1].plot(tr[0, i].real, tr[0, i].imag, "ko", ms=4)
        ax[1].set_title(f"root paths - permutation {cycles}")
        ax[1].legend(fontsize=8)
        for a in ax:
            a.set_aspect("equal")
        fig.tight_layout()
        fig.savefig("monodromy.png", dpi=120)
        print("saved monodromy.png")
