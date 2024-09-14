# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:38:32 2024

@author: amine
"""

from IPython.display import display, Math
import numpy as np
import qutip as qt
import sympy as sp


DECIMALS = 5
TOL = 10 ** (-DECIMALS)

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


def norm(vector: sp.Matrix, subs: dict):
    if not isinstance(vector, sp.Matrix):
        vector = sp.Matrix(vector)
    vector = vector.H.dot(vector)
    return vector.subs(subs).evalf()


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


def vector_to_matrix(vector: sp.Matrix) -> sp.Matrix:
    if vector.shape[1] == 1:
        return vector * vector.H
    else:
        return vector


def unitary_transform(unitary: np.ndarray, state: np.ndarray) -> np.ndarray:
    if len(state.shape) > 1:
        return unitary @ state @ np.conjugate(unitary)
    else:
        return unitary @ state


def align_with_0(q: sp.Matrix):
    return sp.Matrix([[np.conjugate(q[0]), np.conjugate(q[1])], [-q[1], q[0]]])


def Born_rule(projector: sp.Matrix, state: sp.Matrix) -> float:
    return sp.trace(vector_to_matrix(projector) * vector_to_matrix(state))


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


def is_unitary(matrix, tol=TOL):
    """
    Check whether the `matrix` is unitary.

    Parameters
    ----------
    matrix : np.ndarray
        Some complex matrix
    tol : float, optional
        Numerical tolerance

    Returns
    -------
    bool
        True if the product of the matrix and its conjugate transpose is equal
        to the identity matrix.
    """

    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix.evalf(), dtype=complex)

    # Calculate the conjugate transpose (Hermitian).
    conj_transpose = np.conjugate(matrix.T)

    # Perform the multiplication of the matrix with its conjugate transpose.
    product = np.dot(matrix, conj_transpose)

    # Is the product equal to the identity within the specified tolerance?
    return np.allclose(product, np.eye(matrix.shape[0]), atol=tol)


def round(value, decimals=DECIMALS):
    if isinstance(value, np.complex64) or isinstance(value, complex):
        if abs(value.imag) < 10 ** (-decimals):
            value = value.real
    elif isinstance(value, np.ndarray):
        if (abs(value.imag) < 10 ** (-decimals)).all():
            value = value.real
    if value < 10 ** (-decimals):
        return 0

    return np.round(value, decimals)


def purity(rho, tol=TOL):
    """
    Compute the purity of a quantum state `rho`.

    Parameters
    ----------
    rho : np.ndarray
        Quantum state
    tol : float, optional
        Numerical tolerance

    Returns
    -------
    purity : float
        Purity of the quantum state
    """

    if not isinstance(rho, np.ndarray):
        rho = np.array(rho.evalf(), dtype=complex)

    purity = np.trace(rho @ rho)
    assert -tol < trim_imaginary(purity) < 1 + tol, f"purity is {purity}"

    if purity.imag < tol:
        purity = purity.real


def trim_imaginary(matrix, tol=TOL):
    """
    For better readability, trim the imaginary parts when they are negligible
    within numerical tolerance.

    Parameters
    ----------
    matrix :  np.ndarray
        Complex matrix
    tol : float, optional
        Numerical tolerance

    Returns
    -------
    matrix : np.ndarray
        The same matrix with its imaginary part removed if negligible

    """

    if isinstance(matrix, complex) and matrix.imag < tol:
        matrix = matrix.real
    elif (matrix.imag < tol).all():
        matrix = matrix.real

    return matrix

    return purity
