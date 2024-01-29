# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:04:59 2023

@author: Amine Laghaout

Generalization of the Elitzur-Vaidman bomb tester to `N`-partite Mach-Zehnder
interferometers where the bombs can be in a superposition of being on a path
and away from the path.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantum_information as qi
import qutip as qt
import seaborn as sns

ROUND = 4

bomb_dict = {
    "0": np.array([0, 0, 1]),
    "1": np.array([0, 1, 0]),
    "½": np.array([0, 1, 1]) / np.sqrt(2),
    "⅓": np.array([0, 1, np.sqrt(2)]) / np.sqrt(3),
    "⅔": np.array([0, np.sqrt(2), 1]) / np.sqrt(3),
    "h": np.array([0, 1, 1j]) / np.sqrt(2),
    "t": np.array([0, 1, 1j * np.sqrt(2)]) / np.sqrt(3),
    "T": np.array([0, np.sqrt(2), 1j]) / np.sqrt(3),
}


class System:
    def __init__(self, b, g=None):
        # Tuple of bomb states
        if isinstance(b, str):
            self.b = tuple(bomb_dict[bomb] for bomb in b)

        # Number of paths
        self.N = len(self.b)

        # If unspecified, assume that the photon is in a uniform, coherent
        # superposition over all modes.
        if g is None:
            self.g = np.array([0] + [1] * self.N) / np.sqrt(self.N)
        else:
            if isinstance(g, str):
                g = [int(k) for k in list(g)]
            if isinstance(g, list):
                g = np.array(g, dtype=complex)
            assert isinstance(g, np.ndarray)
            if abs(g @ np.conj(g) - 1) > qi.TOL:
                g /= np.sqrt(g @ np.conj(g))
            self.g = g

        # Beam splitter
        self.BS = qi.symmetric_BS(self.N, include_vacuum_mode=True)

    def __call__(self):
        # Coefficients of the basis vectors just before the measurement
        self.compute_coeffs()

        # Initialization of the post-measurement states of the bombs keyed by
        # outcome (i.e., photon click position) and by bomb index (i.e., path
        # location).
        bombs = {
            o: {
                b: np.zeros((2, 2), dtype=complex)
                for b in range(1, self.N + 1)
            }
            for o in range(1, self.N + 1)
        }

        # For each outcome...
        for outcome in range(1, self.N + 1):
            # Retrieve the basis states (ket vectors).
            kets = self.coeffs.index.to_list()

            # For each bomb...
            for b in range(self.N):
                # Compute the start-end pairs for the current outcome. See the
                # manuscript for full details.
                bomb = [
                    (
                        # Start coefficient
                        self.coeffs.loc[kets[k]][outcome],
                        # End coefficient
                        self.coeffs.loc[kets[k + 2 ** (self.N - 1)]][outcome],
                    )
                    for k in range(1, 2 ** (self.N - 1))
                ]
                # Lone term
                bomb += [self.coeffs.loc[kets[2 ** (self.N - 1)]][outcome]]

                # Compute the 2×2 density matrix of the unexploded bomb. Note
                # that this would have been 3×3 if we kept the 0th explosion
                # basis ket, but since the photon reached the detector and
                # therefore bomb remains unexploded, that 0th dimension can be
                # ignored.
                for n, m in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    # Coefficient of ∣n⟩⟨m∣
                    bombs[outcome][b + 1][n, m] = np.sum(
                        [
                            bomb[k][n] * np.conj(bomb[k][m])
                            for k in range(len(bomb) - 1)
                        ]
                    )

                # Add the lone term to ∣1⟩⟨1∣.
                bombs[outcome][b + 1][n, m] += bomb[len(bomb) - 1] * np.conj(
                    bomb[len(bomb) - 1]
                )

                # TODO: Double-check why we're multiplying by the probability.
                bombs[outcome][b + 1] *= self.P[outcome]
                bombs[outcome][b + 1] = qi.trim_imaginary(
                    bombs[outcome][b + 1], qi.TOL
                )

                assert qi.is_density_matrix(bombs[outcome][b + 1])

                # Move one "bit" to the right for the next bomb. Cf. the
                # manuscript for the full details.
                kets = [k[-1] + k[:-1] for k in kets]

        self.bombs = bombs

    def compute_coeffs(self):
        """
        Compute the coefficients of the basis states when there is no
        explosion.
        """

        self.coeffs = pd.DataFrame(index=list(range(2**self.N)))

        # Convert to binary representation.
        self.coeffs["z"] = self.coeffs.index.map(lambda x: bin(x)[2:])

        # Pad with zeros and add one to each "bit" so, e.g., 01 becomes 12,
        # namely a photon on the first path and no photon on the second path.
        self.coeffs["z"] = self.coeffs["z"].apply(
            lambda x: "".join(
                [str(int(y) + 1) for y in list("0" * (self.N - len(x)) + x)]
            )
        )
        self.coeffs.set_index("z", inplace=True)

        # Compute the coefficient for each photon outcome.
        for n in range(1, self.N + 1):
            self.coeffs[n] = self.coeffs.index
            self.coeffs[n] = self.coeffs[n].apply(
                lambda z: self.compute_coeff(n, z)
            )

        self.P = self.coeffs.apply(
            lambda y: qi.trim_imaginary(np.conj(y) @ y, qi.TOL), axis=0
        )

        # Normalize
        self.coeffs /= self.P

    def compute_coeff(self, n, z):
        Z = np.prod([self.b[m][int(x)] for m, x in enumerate(z)])
        Z *= np.multiply(self.BS[1:, n], self.g[1:]) @ np.array(
            [j == "2" for j in z]
        )

        return Z


# %% Run as a script, not as a module.
if __name__ == "__main__":
    system = System("⅓t½", "0111")
    system()
    bombs = system.bombs
    print(np.round(system.P, ROUND))

# %%


def foo(config):
    combis = [
        ("ABC", "111", 1, 1),
        ("AB", "110", -1, 2 / 3),
        ("AC", "101", -1, 2 / 3),
        ("BC", "011", -1, 2 / 3),
        ("A", "100", 1, 1 / 3),
        ("B", "010", 1, 1 / 3),
        ("C", "001", 1, 1 / 3),
    ]
    epsilon = 0
    rho = dict()

    for c in combis:
        system = System(config, "0" + c[1])
        system()
        # c[3] is the prior of a photon click
        epsilon += c[2] * system.P * c[3]
        rho[c[0]] = qi.trim_imaginary(system.bombs[2][1])

    print(f"{config}:\nepsilon =\n{np.round(epsilon, ROUND)}")

    return rho


rho = foo("½½½")  # ⅓t½
# TODO: Check linear combination leading to rho for ABC


def plot_Wigner(bombs=bombs):
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
                f"outcome {o} bomb {b} purity {np.round(qi.purity(bombs[o][b]), ROUND)}"
            )

        plt.show()
