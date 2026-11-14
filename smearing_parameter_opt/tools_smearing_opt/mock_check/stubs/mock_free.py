"""
Laptop stand-in for the QUDA solver: the exact inverse of the FREE Wilson Dirac operator (unit gauge links, antiperiodic
in time), done with a 4D FFT. It is translation invariant and has a Hermitian transfer matrix, so identities that hold
in QCD only on average over configurations hold here exactly. Gammas: PyQUDA's DeGrand-Rossi matrices (pyquda/gamma.py).
"""
import numpy as np

GAMMA = [
    np.eye(4, dtype="<c16"),
    np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]], "<c16"),
    np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], "<c16"),
    np.array([[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]], "<c16"),
    np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], "<c16"),
]


def gamma(n):
    """Gamma(n) = g1^n0 g2^n1 g3^n2 g4^n3, n = 8 n3 + 4 n2 + 2 n1 + n0, as in PyQUDA."""
    return (GAMMA[1] if n & 1 else GAMMA[0]) @ (GAMMA[2] if n & 2 else GAMMA[0]) @ (GAMMA[3] if n & 4 else GAMMA[0]) @ (GAMMA[4] if n & 8 else GAMMA[0])


def free_wilson_inverse(source, mass):
    """source[t, z, y, x, spin, spin, color, color] on the whole lattice -> D^-1 source (D acts on the first spin index)."""
    T, Z, Y, X = source.shape[:4]
    twist = np.exp(-1j * np.pi * np.arange(T) / T).reshape(T, 1, 1, 1, 1, 1, 1, 1)  # antiperiodic in t: p_t = 2 pi (n + 1/2) / T
    b = np.fft.fftn(source * twist, axes=(0, 1, 2, 3))
    p = [2 * np.pi * (np.arange(T) + 0.5) / T, 2 * np.pi * np.arange(Z) / Z, 2 * np.pi * np.arange(Y) / Y, 2 * np.pi * np.arange(X) / X]
    shapes = [(T, 1, 1, 1), (1, Z, 1, 1), (1, 1, Y, 1), (1, 1, 1, X)]
    gammas = [GAMMA[4], GAMMA[3], GAMMA[2], GAMMA[1]]  # direction t, z, y, x
    M = mass + sum((1 - np.cos(pi)).reshape(sh) for pi, sh in zip(p, shapes))
    slash = sum(np.sin(pi).reshape(sh + (1, 1)) * g for pi, sh, g in zip(p, shapes, gammas))  # [t, z, y, x, 4, 4]
    s2 = sum((np.sin(pi) ** 2).reshape(sh) for pi, sh in zip(p, shapes))
    Dinv = (M[..., None, None] * np.eye(4) - 1j * slash) / (M**2 + s2)[..., None, None]
    x = np.einsum("tzyxik,tzyxkjab->tzyxijab", Dinv, b)
    return np.fft.ifftn(x, axes=(0, 1, 2, 3)) / twist


def mock_mass(cfg):
    """One "configuration" = one quark mass, so that different cfgs give different correlators."""
    return 0.30 + 0.02 * ((int(cfg) // 6) % 5)
