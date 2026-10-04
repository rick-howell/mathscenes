"""Episode - the identity element of the sandpile group.

Put grains of sand on the cells of a region of the square grid. A cell holding 4 or
more grains is unstable: it *topples*, sending one grain to each of its 4 neighbours
(grains pushed outside the region fall off the edge for good). Keep toppling until
nothing is unstable; the result does not depend on the order (that's the "abelian"
in abelian sandpile).

Stable configurations that can be reached from *every* configuration by adding sand
are called recurrent. With "add pointwise, then stabilise" (written a ⊕ b) they form
a finite abelian group, the sandpile group. Its size equals the number of spanning
trees of the grid with all outside cells merged into one sink vertex, which is
det Δ for the reduced graph Laplacian Δ (Kirchhoff's matrix-tree theorem).

Every group has an identity e (e ⊕ c = c). A quick recipe for it:

    e = stab( 6 - stab(6) )        where 6 means "6 grains on every cell"

Nobody designed the picture you get; it is just the zero of this group. Here the region
is a disc, so the identity has a disc's symmetry.

Run:  python -m mathscenes.sandpile
"""
import numpy as np


def disc(radius):
    """Boolean mask of the cells within `radius` of the centre (size 2*radius+1)."""
    r = int(radius)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= radius * radius


def topple(h, mask):
    """One parallel sweep: every cell with 4+ grains topples as often as it can.
    Returns (new heights, number of topplings)."""
    k = np.where(mask, h // 4, 0)
    n = int(k.sum())
    if n == 0:
        return h, 0
    h = h - 4 * k
    h[1:, :] += k[:-1, :]
    h[:-1, :] += k[1:, :]
    h[:, 1:] += k[:, :-1]
    h[:, :-1] += k[:, 1:]
    return np.where(mask, h, 0), n


def stabilize(h, mask, keep=None):
    """Topple until stable. keep: sorted sweep numbers to record snapshots at.
    Returns (stable heights, sweeps used, {sweep: (heights, topplings in that sweep)})."""
    h = np.where(mask, h, 0).astype(np.int32)
    snaps, sweep, want = {}, 0, list(keep or [])
    if want and want[0] == 0:
        snaps[0] = (h.copy(), 0)
        want.pop(0)
    while True:
        h, n = topple(h, mask)
        if n == 0:
            break
        sweep += 1
        if want and want[0] == sweep:
            snaps[sweep] = (h.copy(), n)
            want.pop(0)
    for s in want:                                   # anything asked for after the end
        snaps[s] = (h.copy(), 0)
    return h, sweep, snaps


def identity(mask):
    """e = stab(6 - stab(6))."""
    six = 6 * mask.astype(np.int32)
    s, _, _ = stabilize(six, mask)
    e, _, _ = stabilize(six - s, mask)
    return e


def add(a, b, mask):
    """The group operation: a ⊕ b = stab(a + b)."""
    return stabilize(a + b, mask)[0]


def reduced_laplacian(mask):
    """Δ for the cells of `mask` (outside cells merged into the sink, which is removed)."""
    ids = -np.ones(mask.shape, int)
    cells = np.argwhere(mask)
    ids[mask] = np.arange(len(cells))
    L = 4 * np.eye(len(cells), dtype=float)
    for i, (y, x) in enumerate(cells):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx]:
                L[i, ids[yy, xx]] = -1
    return L


def count_recurrent(mask):
    """Brute force on a tiny region: c is recurrent iff c ⊕ e = c."""
    cells = np.argwhere(mask)
    e = identity(mask)
    count = 0
    for code in range(4 ** len(cells)):
        c = np.zeros(mask.shape, np.int32)
        for (y, x) in cells:
            c[y, x] = code % 4
            code //= 4
        if np.array_equal(add(c, e, mask), c):
            count += 1
    return count


if __name__ == "__main__":
    # 1. group order = det Δ (matrix-tree theorem), checked by brute force on small regions
    for shape in [(1, 2), (2, 2), (2, 3), (3, 3)]:
        m = np.ones(shape, bool)
        print(shape, "recurrent:", count_recurrent(m),
              " det Δ:", round(np.linalg.det(reduced_laplacian(m))))
    # 2. the identity on a disc, and its defining property
    m = disc(30)
    e = identity(m)
    print("e ⊕ e == e:", np.array_equal(add(e, e, m), e))
    rng = np.random.default_rng(0)
    c = add(rng.integers(0, 4, m.shape) * m, 3 * m.astype(np.int32), m)   # some recurrent c
    print("e ⊕ c == c:", np.array_equal(add(e, c, m), c))
    sign, logdet = np.linalg.slogdet(reduced_laplacian(m))
    print(f"disc of radius 30: {int(m.sum())} cells, group order = det Δ ≈ 10^{logdet / np.log(10):.0f}")
    for row in e[::3, ::2]:
        print("".join(" .:#"[v] for v in row))
