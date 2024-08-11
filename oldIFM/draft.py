# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:41:44 2024

@author: amine
"""

import numpy as np
import pandas as pd
import quantum_information as qi

g = -1 / 2 + 1j * np.sqrt(3) / 2
gp = 1 / 2 + 1j * np.sqrt(3) / 2
gm = 1 / 2 - 1j * np.sqrt(3) / 2
g2 = -1 / 2 - 1j * np.sqrt(3) / 2

bombs = (
    np.array([np.sqrt(2), 1], dtype=float) / np.sqrt(3),
    np.array([1, 1j], dtype=complex) / np.sqrt(2),
    np.array([1, 1], dtype=float) / np.sqrt(2),
)

# bombs = (
#     np.array([1, 1], dtype=float) / np.sqrt(2),
#     np.array([1, 1j], dtype=complex) / np.sqrt(2),
#     np.array([1, 1], dtype=float) / np.sqrt(2))


psi = (
    np.array(
        [
            [3, 2, 2, 1, 2, 1, 1, 0],
            [0, gp, gm, 1, -1, g, g2, 0],
            [0, gm, gp, 1, -1, g2, g, 0],
        ],
        dtype=complex,
    )
    / 3
)

psi = pd.DataFrame(psi, index=range(1, len(bombs) + 1)).T

bombs = pd.DataFrame(bombs, index=range(1, len(bombs) + 1)).T


def unitary(bomb: np.array):
    return np.array(
        [[np.conjugate(bomb[0]), np.conjugate(bomb[1])], [-bomb[1], bomb[0]]],
        dtype=complex,
    )


U = unitary(bombs[1])
for b in range(2, bombs.shape[1] + 1):
    U = np.kron(U, unitary(bombs[b]))

qi.is_unitary(U)

P = U @ psi[1]
for i, k in enumerate(P):
    i = bin(i)[2:]
    i = "0" * (3 - len(i)) + i
    p = np.conjugate(k) * k
    print(
        f"{i}:",
        np.round(p, 4),
        "-",
        np.real(np.round(k, 4)),
        f"+ j·{np.imag(np.round(k, 4))}"
        if np.imag(np.round(k, 4)) > 1e-4
        else "",
    )
