# mathscenes

The mathematics behind the videos on **Pixelemming** (YouTube, TikTok and Instagram:
@pixelemming), as plain Python + NumPy.

Each file is one video. It only *computes* things (roots, fields, paths, grids, tilings),
so you can read it top to bottom, run it, change the numbers and build your own pictures
on top of it. The top of every file explains the idea in words.

## Get started

```
git clone https://github.com/rick-howell/mathscenes
pip install numpy matplotlib
python -m mathscenes.newton
```

Run the commands from the folder that *contains* `mathscenes/` (the folder you cloned
into). Each file prints its checks: the numbers behind what the video shows.

## Videos

| File | Video | What it computes |
|---|---|---|
| `monodromy.py` | Monodromy | Roots of z⁵ − z + c as c runs a loop around the branch points, and the permutation they come back in (also saves `monodromy.png`) |
| `sandpile.py` | The sandpile identity | Toppling sand on a grid; the identity element e = (6 − 6°)° of the sandpile group; the group's size as a determinant |
| `aztec.py` | The Aztec diamond | Random domino tilings by domino shuffling, the count 2^(n(n+1)/2), and the frozen corners outside the arctic circle |
| `newton.py` | Newton's method | The three basins of z³ = 1 under Newton's step, and how many steps each start takes |
| `tonnetz.py` | The Tonnetz | Notes on a triangular lattice by φ(a, b) = 7a + 4b mod 12; its kernel Λ; chords as triangles; the P, L, R moves round the torus C/Λ |
| `torus.py` | Two ways to build a torus | The square with glued edges, S¹ × S¹, the map between them, and the checks that it is a bijection |
| `norms.py` | The parallelogram law (video coming) | p-norms and when ‖x + y‖² + ‖x − y‖² = 2‖x‖² + 2‖y‖² holds |
| `svf.py` | The channel's banner | A state variable filter's transfer functions (low-, band-, high-pass, notch), their poles and zeros |

## Shared tools (`common.py`)

- `complex_grid`: a grid of complex numbers to evaluate a function on
- `level_lines`: where |f| and arg f take "round" values (the contour lines in the videos)
- `track_continuously`: follow unordered points (like polynomial roots) along a path
- `permutation_cycles`: how a closed path permutes those points

## Ideas to try

- `monodromy.py`: change the loop in `figure_eight()`. A circle around one branch point
  swaps two roots; a big circle around all four gives a 5-cycle.
- `newton.py`: try z⁴ = 1 or z³ − 2z + 2 and look for starts that never settle.
- `tonnetz.py`: change 7a + 4b to another pair of intervals and see which lattice you get.
- `svf.py`: raise Q and watch the poles move towards the imaginary axis.

## How this was made

The topics, the choice of what to show, the look and the final review are mine. The code
was written with the help of an AI assistant (Claude), and the maths in each file is
checked by the tests in its `__main__`. The videos are rendered by code, frame by frame,
with synthesized music; no image, video or audio generators are used.

## Licence

MIT (see `LICENSE`): use, change and share the code freely, keeping the copyright notice.
