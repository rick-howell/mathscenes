"""Episode - the Aztec diamond and the arctic circle.

The Aztec diamond of order n is the staircase-shaped region of unit squares whose centres
satisfy |x| + |y| <= n. It can be tiled by 2×1 dominoes in exactly 2^(n(n+1)/2) ways
(Elkies, Kuperberg, Larsen and Propp, 1992).

Pick one of those tilings uniformly at random. For large n something unexpected
happens: near the four corners the dominoes all line up in brickwork ("frozen"), and the
disorder is confined to a disc. Scaled so the diamond is |x| + |y| <= 1, the boundary
between the two is the inscribed circle x² + y² = 1/2: the arctic circle theorem
(Jockusch, Propp and Shor, 1995).

How to sample a uniform tiling: domino shuffling. Give every domino a direction
(N, S, E, W) and grow the diamond one order at a time:
  1. destroy every 2×2 block whose two dominoes are about to move into each other,
  2. slide every remaining domino one step in its direction,
  3. the holes are now 2×2 blocks: fill each with two horizontal dominoes (top one N,
     bottom one S) or two vertical ones (left W, right E), with a fair coin.
After n steps the tiling is uniform among all tilings of the order-n diamond.

Run:  python -m mathscenes.aztec
"""
import numpy as np

N, S, E, W = 0, 1, 2, 3                     # direction each domino moves in the next step
MOVE = {N: (0, 1), S: (0, -1), E: (1, 0), W: (-1, 0)}


def cells(d):
    """The two unit squares (lower-left corners) of a domino (x, y, kind)."""
    x, y, k = d
    return ((x, y), (x + 1, y)) if k in (N, S) else ((x, y), (x, y + 1))


def in_diamond(x, y, n):
    return abs(x + 0.5) + abs(y + 0.5) <= n


def step(dominoes, n, rng):
    """Grow a tiling of order n into a uniformly random tiling of order n + 1.
    Returns (new dominoes, survivors before moving, destroyed, created)."""
    at = {(x, y, k) for (x, y, k) in dominoes}
    bad = set()
    for (x, y, k) in dominoes:
        if k == N and (x, y + 1, S) in at:          # N below S: they would collide
            bad |= {(x, y, N), (x, y + 1, S)}
        if k == E and (x + 1, y, W) in at:          # E left of W
            bad |= {(x, y, E), (x + 1, y, W)}
    survivors = [d for d in dominoes if d not in bad]
    moved = [(x + MOVE[k][0], y + MOVE[k][1], k) for (x, y, k) in survivors]
    filled = set()
    for d in moved:
        filled.update(cells(d))
    created = []
    m = n + 1
    for y in range(-m, m):                          # holes, scanned bottom-up, left-right
        for x in range(-m, m):
            if (x, y) in filled or not in_diamond(x, y, m):
                continue
            if rng.random() < 0.5:                  # two horizontal: bottom S, top N
                block = [(x, y, S), (x, y + 1, N)]
            else:                                   # two vertical: left W, right E
                block = [(x, y, W), (x + 1, y, E)]
            for d in block:
                filled.update(cells(d))
            created += block
    return moved + created, survivors, sorted(bad), created


def random_tiling(n, seed=0, history=False):
    """Uniformly random tiling of the order-n Aztec diamond by domino shuffling.
    history=True also returns, for each step k -> k+1, (survivors, destroyed, created)."""
    rng = np.random.default_rng(seed)
    dom, hist = [], []
    for k in range(n):
        dom, surv, bad, new = step(dom, k, rng)
        if history:
            hist.append((surv, bad, new))
    return (dom, hist) if history else dom


def grid(dominoes, n):
    """2n×2n array of domino kinds (−1 outside the diamond); row 0 is the top."""
    g = -np.ones((2 * n, 2 * n), int)
    for d in dominoes:
        for (x, y) in cells(d):
            g[n - 1 - y, x + n] = d[2]
    return g


def is_tiling(dominoes, n):
    seen = set()
    for d in dominoes:
        for c in cells(d):
            if c in seen or not in_diamond(*c, n):
                return False
            seen.add(c)
    return len(seen) == 2 * n * (n + 1)


def count_tilings(n):
    """Brute-force count (small n only), to check 2^(n(n+1)/2)."""
    region = sorted({(x, y) for x in range(-n, n) for y in range(-n, n) if in_diamond(x, y, n)},
                    key=lambda c: (c[1], c[0]))
    free = set(region)

    def go():
        if not free:
            return 1
        x, y = min(free, key=lambda c: (c[1], c[0]))
        total = 0
        for other in ((x + 1, y), (x, y + 1)):
            if other in free:
                free.difference_update({(x, y), other})
                total += go()
                free.update({(x, y), other})
        return total
    return go()


def frozen_fraction(g, n, r_lo, r_hi):
    """Fraction of cells with r_lo <= |·|/n < r_hi whose domino points 'outwards' along the
    nearest corner, i.e. the brickwork of the frozen regions."""
    rows, cols = np.nonzero(g >= 0)
    x = cols - n + 0.5
    y = (n - 1 - rows) + 0.5
    r = np.hypot(x, y) / n
    sel = (r >= r_lo) & (r < r_hi)
    kind = g[rows, cols]
    corner = np.where(np.abs(y) >= np.abs(x), np.where(y > 0, N, S), np.where(x > 0, E, W))
    return float((kind[sel] == corner[sel]).mean())


if __name__ == "__main__":
    print("number of tilings, brute force vs 2^(n(n+1)/2):")
    for n in range(1, 5):
        print(f"  n = {n}: {count_tilings(n)} vs {2 ** (n * (n + 1) // 2)}")
    from collections import Counter
    c = Counter(tuple(sorted(random_tiling(2, seed=s))) for s in range(8000))
    print(f"order 2, 8000 samples: {len(c)} distinct tilings, counts {sorted(c.values())}")
    dom = random_tiling(150, seed=1)
    print("order 150 is a tiling:", is_tiling(dom, 150))
    g = grid(dom, 150)
    for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.68), (0.74, 0.8), (0.8, 1.0)):
        print(f"  |x|/n in [{lo}, {hi}): frozen fraction {frozen_fraction(g, 150, lo, hi):.3f}")
    print("(the arctic circle sits at |x|/n = 1/√2 ≈ 0.707)")
