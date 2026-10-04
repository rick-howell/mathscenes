"""The state variable filter (SVF): one circuit, four filters, two poles.

A second-order state variable filter has a cutoff ω0 and a resonance Q. Its outputs, as
transfer functions of the complex frequency s, all share the same denominator

    D(s) = s² + (ω0/Q) s + ω0²,

whose two roots are the poles  s = ω0 ( -1/(2Q) ± i √(1 - 1/(4Q²)) )  (for Q > 1/2):
a conjugate pair in the left half-plane, on the circle |s| = ω0. The outputs differ only
in their zeros:

    low-pass   H_LP = ω0² / D          no zeros
    band-pass  H_BP = (ω0/Q) s / D     a zero at s = 0
    high-pass  H_HP = s² / D           a double zero at 0
    notch      H_N  = (s² + ω0²) / D   zeros at ±iω0, on the imaginary axis

On the imaginary axis s = iω, |H(iω)| is the frequency response you hear. Raising Q
pulls the poles towards the axis, which is the resonant peak.

Run:  python -m mathscenes.svf
"""
import numpy as np


def D(s, w0=1.0, Q=2.0):
    return s * s + (w0 / Q) * s + w0 * w0


def H(s, kind="lowpass", w0=1.0, Q=2.0):
    num = {"lowpass": w0 * w0, "bandpass": (w0 / Q) * s, "highpass": s * s,
           "notch": s * s + w0 * w0}[kind]
    return num / D(s, w0, Q)


def poles(w0=1.0, Q=2.0):
    re, im = -w0 / (2 * Q), w0 * np.sqrt(max(0.0, 1 - 1 / (4 * Q * Q)))
    return np.array([re + 1j * im, re - 1j * im])


def zeros(kind="lowpass", w0=1.0):
    return {"lowpass": np.array([]), "bandpass": np.array([0j]), "highpass": np.array([0j, 0j]),
            "notch": np.array([1j * w0, -1j * w0])}[kind]


if __name__ == "__main__":
    p = poles(1.0, 2.0)
    print("poles (w0 = 1, Q = 2):", np.round(p, 4), " |p| =", np.round(np.abs(p), 4))
    print("D vanishes there:", np.allclose(D(p), 0))
    w = np.array([0.1, 0.5, 0.97, 1.0, 2.0, 10.0])
    for kind in ("lowpass", "bandpass", "highpass", "notch"):
        print(f"{kind:9s} |H(iw)| at w = {w.tolist()}:", np.round(np.abs(H(1j * w, kind)), 3))
    print("peak of the low-pass response is near w0 with height ~Q:",
          round(float(np.abs(H(1j * 0.97, 'lowpass'))), 3))
