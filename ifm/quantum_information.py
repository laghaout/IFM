# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:57:06 2023

@author: Amine Laghaout
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import qutip as qt

# Numerical tolerance
DEC = 13
TOL = 10 ** (-DEC)


def BS_op(BS, rho):
    """
    Operate the beam splitter `BS` on a density matrix `rho`.

    Parameters
    ----------
    BS : np.ndarray
        Beam splitter matrix
    rho : np.ndarray
        Density matrix

    Returns
    -------
    rho : np.ndarray
        Density matrix of the transformed quantum state
    """

    rho = BS @ rho @ np.conjugate(BS.T)

    return rho


def symmetric_BS(N, include_vacuum_mode=True):
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
    include_vacuum_mode : bool, optional
        Include the possibility that only the vacuum is "incident" on the beam
        splitter?

    Returns
    -------
    BS : np.dnarray
        Matrix representation of the beam splitter with at most photon
        excitation
    """

    a = 2 * np.pi / N
    a = np.cos(a) + 1j * np.sin(a)
    BS = np.array([[a ** (r * c) for c in range(N)] for r in range(N)])
    if include_vacuum_mode:
        BS = np.concatenate((np.zeros((N, 1)), BS), axis=1)
        BS = np.concatenate((np.zeros((1, N + 1)), BS), axis=0)
        BS[0, 0] = np.sqrt(N)
    BS /= np.sqrt(N)

    return trim_imaginary(BS)


def plot_density_matrix(rho, title=None):
    """
    Plot the density matrix `rho`.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix
    title : str, optional
        Title for the plot

    Returns
    -------
    None.
    """

    cells = rho.shape[0]

    # Use Matplotlib to render the array with a grid
    fig, ax = plt.subplots()

    # TODO: Also represent the imaginary part
    assert False
    cax = ax.matshow(rho.real, cmap="viridis")

    # Add color bar
    plt.colorbar(cax)

    # Set ticks
    ax.set_xticks(np.arange(-0.5, cells, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cells, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.08)

    # Hide major tick labels
    ax.tick_params(
        which="major",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    if not isinstance(title, None):
        plt.title(title)

    plt.show()


def is_density_matrix(rho, tol=TOL):
    """
    Check that the matrix `rho` is positive semi-definite and of trace one.

    Parameters
    ----------
    rho : np.ndarray
        Some complex matrix
    tol : float
        Numerical tolerance

    Returns
    -------
    bool
        True if the `rho` has
        - trace 1, and
        - all its eigenvalues are positive.
    """

    return abs(1 - np.trace(rho)) < tol and np.all(
        np.linalg.eigvalsh(rho) >= -tol
    )


def is_Hermitian(matrix, tol=TOL):
    """
    Check that the `matrix` is Hermitian.

    Parameters
    ----------
    matrix : np.ndarray
        Some complex matrix
    tol : float, optional
        Numerical tolerance

    Returns
    -------
    bool
        True if the `matrix` is Hermitian
    """

    return np.allclose(matrix, np.conjugate(matrix.T), atol=tol)


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

    # Calculate the conjugate transpose (Hermitian).
    conj_transpose = np.conjugate(matrix.T)

    # Perform the multiplication of the matrix with its conjugate transpose.
    product = np.dot(matrix, conj_transpose)

    # Is the product equal to the identity within the specified tolerance?
    return np.allclose(product, np.eye(matrix.shape[0]), atol=tol)


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

    purity = np.trace(rho @ rho)
    assert -tol < purity < 1 + tol, f"purity is {purity}"

    if purity.imag < tol:
        purity = purity.real

    return purity


def Born(P, rho, tol=TOL):
    """
    Apply Born's rule using a projection operator `P` and a quantum state
    `rho`.

    Parameters
    ----------
    P : np.ndarray
        Projection operator
    rho : np.ndarray
        Quantum state
    tol : float
        Numerical tolerance

    Returns
    -------
    p : float
        Probability of projection of `rho` onto `P`

    """

    p = np.trace(P @ rho)
    assert -tol < p < 1 + tol

    if p.imag < tol:
        p = p.real

    return p


def partial_trace(rho, keep, dims, optimize=False):
    """
    Calculate the partial trace. See:
    https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array of int
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array of int
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """

    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)

    return rho_a.reshape(Nkeep, Nkeep)


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

    if (matrix.imag < tol).all():
        matrix = matrix.real

    return matrix


# %% Clean below


def sqrt_matrix(a):
    evalues, evectors = np.linalg.eig(a)
    # Ensuring square root matrix exists
    assert (evalues >= 0).all()
    return evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)


def fidelity(a, b):
    # np.trace(a, b)

    pass


def reload(
    results_dir=["..", "..", "..", "..", "physics", "Larsson", "results"]
):
    return pickle.load(
        open(os.path.join(*results_dir + ["results.pkl"]), "rb")
    )


def results_vs_N(results, main=1, outcome=2, subsystem="bomb 1"):
    df = pd.DataFrame(
        {
            "config": results.keys(),
        }
    ).set_index("config")
    df["N"] = df.index.to_series().apply(lambda x: len(x))
    df["purity"] = df.index.to_series().apply(
        lambda x: results[x][outcome]["subsystems"][subsystem]["purity"]
    )
    df["present"] = df.index.to_series().apply(
        lambda x: results[x][outcome]["subsystems"][subsystem]["diagonals"][1]
    )
    df["main purity"] = df.index.to_series().apply(
        lambda x: results[x][main]["subsystems"][subsystem]["purity"]
    )
    df["main present"] = df.index.to_series().apply(
        lambda x: results[x][main]["subsystems"][subsystem]["diagonals"][1]
    )

    return df


def get_subrho(config, outcome, bomb, results, rounding=DEC):
    if rounding is None:
        return results[config][outcome]["subsystems"][bomb]["subrho"]
    elif isinstance(rounding, int):
        return np.round(
            results[config][outcome]["subsystems"][bomb]["subrho"], DEC
        )


def predicted_purity(undisturbed, disturbed, N, k=2):
    C = math.comb(N, k)

    a_f = N - 1
    a_i = C - N + 1

    # print("a_i:", a_i, "a_f:", a_f, "C:", C)

    return purity((a_f * disturbed + a_i * undisturbed) / C)


def reconstruct_disturbed(N, final, undisturbed, k=2):
    C = math.comb(N, k)

    rho_k = (C * final + (N - 1 - C) * undisturbed) / (N - 1)

    return np.round(rho_k, DEC)


def plot_Wigner(bombs, system, ROUND=4):
    xvec = np.linspace(-5, 5, 200)
    # rho_coherent = qt.coherent_dm(N, np.sqrt(2))
    for o in bombs.keys():
        rho_bombs = {b: qt.Qobj(bombs[o][b]) for b in bombs[o].keys()}

        wigner_bombs = {
            b: qt.wigner(rho_bombs[b], xvec, xvec) for b in bombs[o].keys()
        }

        # Plot the results
        fig, axes = plt.subplots(1, system.N, figsize=(4 * system.N, system.N))
        cont = [None] * system.N
        lbl = [None] * system.N
        for b in range(1, system.N + 1):
            cont[b - 1] = axes[b - 1].contourf(
                xvec, xvec, wigner_bombs[b], 100
            )
            lbl[b - 1] = axes[b - 1].set_title(
                f"outcome {o} bomb {b} purity {np.round(purity(bombs[o][b]), ROUND)}"
            )

        plt.show()
