"""Episode - the Tonnetz is a torus.

1. Notes as numbers. Count in semitones: n in Z. Notes an octave apart sound "the same",
   so the note is n mod 12: the map Z -> Z/12Z wraps the number line onto a clock of
   12 pitch classes (0 = C, 1 = C#, ..., 11 = B).

2. A plane of notes. On the triangular lattice Z² ⊂ C (points a + bω, ω = e^{iπ/3}), give
   the point (a, b) the note
                         φ(a, b) = 7a + 4b  mod 12.
   One step right is a perfect fifth (+7), one step up-right a major third (+4), and the
   third edge of each small triangle is a minor third (+3). So every small triangle is a
   chord: pointing up, a major triad (x, x+4, x+7); pointing down, a minor triad. Two
   triangles that share an edge share two notes; flipping across the edge is one of the
   three "neo-Riemannian" moves P, L, R.

3. The kernel. φ is a homomorphism Z² -> Z/12Z onto all 12 notes, so its kernel
   Λ = {(a, b) : 7a + 4b ≡ 0 mod 12} - every point labeled C - is a sublattice of index
   12, spanned for example by (0, 3) (three major thirds = an octave) and (4, -1).
   Hence Z²/Λ ≅ Z/12Z: one tile of Λ holds each note exactly once.

4. The torus. The labeling repeats along Λ, so the Tonnetz is really the plane with
   points identified when they differ by Λ: C/Λ, a torus. Walking P, L, P, L, ... goes
   once around it and comes back to the starting chord after six steps.

Run:  python -m mathscenes.tonnetz
"""
import numpy as np

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
OMEGA = np.exp(1j * np.pi / 3)
V1, V2 = (0, 3), (4, -1)                    # a basis of the kernel Λ


def note(a, b):
    """φ(a, b) = 7a + 4b mod 12."""
    return (7 * np.asarray(a) + 4 * np.asarray(b)) % 12


def point(a, b):
    """The lattice point (a, b) as a complex number a + bω."""
    return a + b * OMEGA


def lattice_coords(z):
    """Inverse of point(): real (a, b) with z = a + bω."""
    b = np.imag(z) / np.sin(np.pi / 3)
    return np.real(z) - b / 2, b


# triangles: ("up", a, b) has corners (a,b), (a+1,b), (a,b+1);
#            ("down", a, b) has corners (a+1,b), (a,b+1), (a+1,b+1)
def corners(tri):
    kind, a, b = tri
    return [(a, b), (a + 1, b), (a, b + 1)] if kind == "up" else [(a + 1, b), (a, b + 1), (a + 1, b + 1)]


def chord(tri):
    """(root pitch class, 'major'/'minor') of a triangle."""
    kind, a, b = tri
    if kind == "up":
        return int(note(a, b)), "major"
    return int(note(a + 1, b) + 9) % 12, "minor"      # (x+4, x+7, x+11) is minor on x+4


def chord_name(tri):
    root, q = chord(tri)
    return NAMES[root] + ("" if q == "major" else "m")


def pitch_classes(tri):
    return sorted(int(note(*c)) for c in corners(tri))


def flip(tri, move):
    """Neo-Riemannian moves as flips across an edge: P keeps the fifth, R keeps the major
    third of a major chord (minor third of a minor), L keeps the other third."""
    kind, a, b = tri
    if kind == "up":            # major chord on (a, b)
        return {"P": ("down", a, b - 1),        # across the fifth edge (a,b)-(a+1,b)
                "L": ("down", a, b),            # across (a+1,b)-(a,b+1): keeps the 3rd and 5th
                "R": ("down", a - 1, b)}[move]  # across (a,b)-(a,b+1): keeps root and 3rd
    return {"P": ("up", a, b + 1),
            "L": ("up", a, b),
            "R": ("up", a + 1, b)}[move]


def walk(start, moves):
    path = [start]
    for m in moves:
        path.append(flip(path[-1], m))
    return path


def in_kernel(a, b):
    return note(a, b) == 0


if __name__ == "__main__":
    print("Z -> Z/12Z:", [f"{n}->{NAMES[n % 12]}" for n in range(10, 16)])
    a, b = np.meshgrid(np.arange(-12, 13), np.arange(-12, 13))
    print("one step right = +7 (fifth), up-right = +4 (major third):",
          note(1, 0), note(0, 1), " third edge (down-right) = +3 (minor third):", (note(1, 0) - note(0, 1)) % 12)
    det = V1[0] * V2[1] - V1[1] * V2[0]
    print("kernel basis", V1, V2, "in kernel:", in_kernel(*V1), in_kernel(*V2), " index |det| =", abs(det))
    # every point labeled C is an integer combination of V1, V2
    ker = [(x, y) for x, y in zip(a.ravel(), b.ravel()) if in_kernel(x, y)]
    M = np.array([V1, V2]).T
    ok = all(np.allclose(np.round(np.linalg.solve(M, k)), np.linalg.solve(M, k)) for k in ker)
    print(f"all {len(ker)} C-points in the window are in span(V1, V2):", ok)
    up = ("up", 0, 0)
    print("triangle up(0,0):", chord_name(up), pitch_classes(up))
    for m in "PLR":
        t = flip(up, m)
        print(f"  {m}: {chord_name(t)} {pitch_classes(t)}  shares {sorted(set(pitch_classes(up)) & set(pitch_classes(t)))}")
    path = walk(up, "PLPLPL")
    print("P L P L P L from C:", [chord_name(t) for t in path])
    print("back to the same chord:", chord_name(path[-1]) == chord_name(path[0]),
          " lattice offset:", (path[-1][1] - path[0][1], path[-1][2] - path[0][2]),
          "in kernel:", in_kernel(path[-1][1] - path[0][1], path[-1][2] - path[0][2]))
