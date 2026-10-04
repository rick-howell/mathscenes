"""Shared numerical tools for complex-function scenes."""
import numpy as np


def complex_grid(width, height, center=0j, span=3.0):
    """A height x width grid of complex numbers covering `span` units horizontally.

    Row 0 is the top of the picture (largest imaginary part), as in an image.
    """
    span_y = span * height / width
    x = np.linspace(center.real - span / 2, center.real + span / 2, width)
    y = np.linspace(center.imag + span_y / 2, center.imag - span_y / 2, height)
    return x[None, :] + 1j * y[:, None]


def level_lines(f, modulus_density=1.0, phase_count=8, width=0.55):
    """Contour structure of a complex field f, measured in *pixels of the grid*.

    Returns a dict of arrays, all the same shape as f:
      log_mod      log2 |f| * modulus_density   (integer values = modulus contours)
      phase        arg f / (2 pi), in [0, 1)
      mod_line     ~1 on the contours log_mod = integer, falling off with distance
      phase_line   ~1 on the rays arg f = k / phase_count
    Distances are divided by the local gradient, so lines come out about `width`
    grid-cells wide everywhere, however steep the function is.
    """
    phase = np.angle(f) / (2 * np.pi)
    log_mod = np.log2(np.abs(f) + 1e-300) * modulus_density
    gy, gx = np.gradient(log_mod)
    d_mod = np.abs(log_mod - np.round(log_mod)) / (np.hypot(gx, gy) + 1e-12)
    # phase gradient from ratios of neighbors (immune to the branch cut)
    px = np.pad(np.angle(f[:, 1:] / f[:, :-1]), ((0, 0), (0, 1)), mode="edge")
    py = np.pad(np.angle(f[1:, :] / f[:-1, :]), ((0, 1), (0, 0)), mode="edge")
    g_ph = np.hypot(px, py) * phase_count / (2 * np.pi) + 1e-12
    ph = phase * phase_count
    d_ph = np.abs(ph - np.round(ph)) / g_ph
    return {
        "log_mod": log_mod,
        "phase": phase % 1.0,
        "mod_line": np.exp(-(d_mod / width) ** 2),
        "phase_line": np.exp(-(d_ph / width) ** 2),
    }


def track_continuously(sets):
    """Given a sequence of unordered point sets (e.g. polynomial roots at successive
    parameter values), reorder each set so every point follows a continuous path.

    sets: array (steps, n) of complex numbers. Returns array of the same shape where
    column i is one continuous track. Uses greedy nearest-neighbor matching, which is
    reliable as long as the steps are small compared with the gaps between points.
    """
    sets = np.asarray(sets)
    out = np.empty_like(sets)
    out[0] = sets[0]
    for k in range(1, len(sets)):
        prev, cur = out[k - 1], sets[k]
        free = list(range(len(cur)))
        row = np.empty_like(cur)
        # match the most constrained points first
        order = np.argsort([np.sort(np.abs(cur - p))[0] for p in prev])
        for i in order:
            j = min(free, key=lambda m: abs(cur[m] - prev[i]))
            row[i] = cur[j]
            free.remove(j)
        out[k] = row
    return out


def permutation_cycles(start, end):
    """How a closed path permutes a set of points: end[i] sits where start[perm[i]] was.
    Returns (perm, cycles) with cycles written 1-based, fixed points omitted."""
    perm = [int(np.argmin(np.abs(start - e))) for e in end]
    seen, cycles = set(), []
    for i in range(len(perm)):
        if i in seen:
            continue
        cyc, j = [], i
        while j not in seen:
            seen.add(j)
            cyc.append(j + 1)
            j = perm[j]
        if len(cyc) > 1:
            cycles.append(cyc)
    return perm, cycles
