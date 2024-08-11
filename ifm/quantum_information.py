# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:38:32 2024

@author: amine
"""

from IPython.display import display, Math
import numpy as np
import qutip as qt
import sympy as sp


DECIMALS = 4

# Dictionary of shorthand notations for "common" qubits
QUBITS_DICT = {
    "½": sp.Matrix([1, 1]) / sp.sqrt(2),
    "i": sp.Matrix([1, sp.I]) / sp.sqrt(2),
    "j": sp.Matrix([1, -sp.I]) / sp.sqrt(2),
    "0": sp.Matrix([1, 0]),
    "1": sp.Matrix([0, 1]),
    "⅓": sp.Matrix([1, sp.sqrt(2)]) / sp.sqrt(3),
    "⅔": sp.Matrix([sp.sqrt(2), 1]) / sp.sqrt(3),
}


def pure_qubit(
    q0: str | complex, q1: str | complex = None, to_qutip: bool = False
):
    # Shorthand
    if isinstance(q0, str) and q1 is None:
        q = QUBITS_DICT[q0]
    # Symbolic
    elif isinstance(q0, str) and isinstance(q1, str):
        q0, q1 = sp.symbols(f"{q0} {q1}")
        q = sp.Matrix([q0, q1])
    # Explicit
    else:
        q = sp.Matrix([q0, q1])

    # TODO: Does this work with symbolic qubits?
    # assert np.allclose(complex(np.array((q.T @ sp.conjugate(q))[0].evalf())), 1)

    return q


def subs(x, **kwargs):
    x = sp.Matrix(x).subs(kwargs)
    x = sp.lambdify([], x, modules="numpy")
    x = x()

    return x


def disp(x):
    display(
        Math(
            sp.latex(sp.Matrix(x).applyfunc(sp.factor).applyfunc(sp.simplify))
        )
    )


def vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    if len(vector.shape) == 1:
        return np.outer(vector, np.conjugate(vector))
    else:
        return vector


def unitary_transform(unitary: np.ndarray, state: np.ndarray) -> np.ndarray:
    if len(state.shape) > 1:
        return unitary @ state @ np.conjugate(unitary)
    else:
        return unitary @ state


def align_with_0(q: sp.Matrix):
    return sp.Matrix([[np.conjugate(q[0]), np.conjugate(q[1])], [-q[1], q[0]]])


def Born(projector, state) -> float:
    return np.trace(vector_to_matrix(projector) @ vector_to_matrix(state))


def symmetric_BS(N):
    """
    Matrix representation of a symmetric `N`-partite beam splitter as per
    - https://doi.org/10.1103/PhysRevA.55.2564
    - https://physics.stackexchange.com/questions/789867/what-is-a-general-expression-for-a-symmetric-n-mode-beam-splitter
    - https://physics.stackexchange.com/questions/787593/what-does-a-beam-splitter-look-like-in-path-encoded-notation-for-at-most-one-pho

    The first dimension of the Hilbert space represents the vacuum whereas all
    subsequent dimensions represent the excitation of a single photon in the
    corresponding port of the beam splitter.

    Parameters
    ----------
    N : int
        Number of modes

    Returns
    -------
    BS : np.dnarray
        Matrix representation of the beam splitter with at most photon
        excitation
    """

    matrix = sp.Matrix(
        [
            [sp.exp(2 * sp.I * sp.pi / N) ** (i * j) for j in range(0, N)]
            for i in range(0, N)
        ]
    )
    matrix /= sp.sqrt(N)

    return matrix


def round(value, decimals=DECIMALS):
    if isinstance(value, complex):
        if abs(value.imag) < 10 ** (-decimals):
            value = value.real
    elif isinstance(value, np.ndarray):
        if (abs(value.imag) < 10 ** (-decimals)).all():
            value = value.real

    return np.round(value, decimals)
