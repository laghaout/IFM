# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:04:59 2023

@author: Amine Laghaout

Generalization of the Elitzur-Vaidman bomb tester to `N`-partite Mach-Zehnder
interferometers where the bombs can be in a superposition of being on a path
and away from the path.
"""

import numpy as np
import pandas as pd
import quantum_information as qi

from itertools import combinations

ROUND = 4  # Number of decimal points to display when rounding

# Short hand for representing the quantum states of the bombs as characters
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
            self.bomb_config = b
            self.b = tuple(bomb_dict[bomb] for bomb in b)
        else:
            self.b = b

        # Infer the number of paths
        self.N = len(self.b)

        # If unspecified, assume that the photon is in a uniform, coherent
        # superposition over all modes.
        if g is None:
            g = np.array([0] + [1] * self.N) / np.sqrt(self.N)
        # If specified...
        else:
            # as a string, convert it to a list of integer, and...
            if isinstance(g, str):
                g = np.array([int(k) for k in list(g)], dtype=complex)
            # if specified as a list, convert it to a NumPy array.
            if isinstance(g, list):
                g = np.array(g, dtype=complex)
            assert isinstance(g, np.ndarray)

            # Normalize the pure quantum state of the photon
            if abs(g @ np.conj(g) - 1) > qi.TOL:
                g /= np.sqrt(g @ np.conj(g))
        self.g = g

        # Beam splitter
        self.BS = qi.symmetric_BS(self.N, include_vacuum_mode=True)

    def __call__(self):
        # Post-photon-measurement state
        self.compute_coeffs()

        # Initialization of the post-photon-measurement states of the bombs
        # keyed by outcome (i.e., photon click position) and by bomb index
        # (i.e., path location).
        bombs = {
            outcome: {  # Outcome
                bomb: np.nan  # Bomb index (path location)
                * np.ones((2, 2), dtype=complex)
                for bomb in range(1, self.N + 1)
            }
            for outcome in range(1, self.N + 1)
        }

        # For each outcome...
        for outcome in range(1, self.N + 1):
            # retrieve the post-measurement state (ket vectors).
            kets = self.coeffs.index.to_list()

            # For each bomb...
            for b in range(self.N):
                # compute the (start, end) pairs for the current outcome. See
                # the manuscript for full details.
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
                # that this would have been 3×3 if we kept the 0th, "explosion"
                # basis ket. However, since the photon reached the detector and
                # the bomb remained unexploded, that 0th dimension can thus be
                # ignored.
                for n, m in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    # Coefficient of ∣n⟩⟨m∣
                    bombs[outcome][b + 1][n, m] = np.sum(
                        [
                            bomb[k][n] * np.conj(bomb[k][m])
                            for k in range(len(bomb) - 1)
                        ],
                        dtype=complex,
                    )

                # Add the lone term to ∣1⟩⟨1∣.
                bombs[outcome][b + 1][n, m] += bomb[len(bomb) - 1] * np.conj(
                    bomb[len(bomb) - 1]
                )

                # Assert that the current comb state is a density matrix.
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

        TODO: Add the 0th, "explosion" outcome.
        """

        # There are 2^N basis states.
        self.coeffs = pd.DataFrame(index=list(range(2**self.N)))

        # Convert to binary representation.
        self.coeffs["ket"] = self.coeffs.index.map(lambda x: bin(x)[2:])

        # Pad with zeros and add one to each "bit" so, e.g., 01 becomes 12,
        # clearedly a photon on the first path and no photon on the second path.
        self.coeffs["ket"] = self.coeffs["ket"].apply(
            lambda x: "".join(
                [str(int(y) + 1) for y in list("0" * (self.N - len(x)) + x)]
            )
        )
        self.coeffs.set_index("ket", inplace=True)

        # Compute the coefficient for each photon outcome.
        for outcome in range(1, self.N + 1):
            self.coeffs[outcome] = self.coeffs.index
            self.coeffs[outcome] = self.coeffs[outcome].apply(
                lambda ket: np.prod(
                    [self.b[m][int(x)] for m, x in enumerate(ket)]
                )
                * np.multiply(self.BS[1:, outcome], self.g[1:])
                @ np.array([j == "2" for j in ket])
            )

        # Compute the probability of each outcome.
        self.P = self.coeffs.apply(
            lambda y: qi.trim_imaginary(np.conj(y) @ y, qi.TOL), axis=0
        )

        # Normalize.
        self.coeffs /= np.sqrt(self.P)

    @staticmethod
    def Born_decomposition(N, k=2):
        """
        Decompose the modes as per Born's rule.

        Parameters
        ----------
        N : int
            Number of modes
        k : cardinality of the decomposition, optional
            This is by default for Born's rule

        Returns
        -------
        df : pandas.DataFrame
            Decomposition of the paths

        """

        # List all the possible paths.
        S = [str(s) for s in range(1, N + 1)]

        # Generate n-choose-k combinations
        comb = combinations(S, k)

        # Convert iterator to a list and print
        comb_list = ["".join([str(k) for k in range(1, N + 1)])]
        comb_list += S + ["".join(sorted(c)) for c in comb]

        mydict = {f"{k}": 0 for k in comb_list}
        mydict["".join([str(k) for k in range(1, N + 1)])] = 1

        for n in range(1, N + 1):
            # print(f"P{n}")
            mydict[f"{n}"] -= 1
            for m in range(1, n):
                # print(f"P{m}{n}-P{n}-P{m}")
                mydict[f"{m}{n}"] -= 1
                mydict[f"{n}"] += 1
                mydict[f"{m}"] += 1

        df = pd.DataFrame(index=mydict.keys())
        # df['cleared'] = df.index.map(lambda x: '0'*N)
        df["cleared"] = df.index.map(
            lambda x: "".join(
                ["1" if str(j) in x else "0" for j in range(1, N + 1)]
            )
        )
        df["weight"] = df.index.map(lambda x: mydict[x])
        df["prior"] = df.index.map(lambda x: len(x) / N)

        return df

    def decompose(self):
        N = self.N

        self.combis = self.Born_decomposition(N)
        self.combis[[n for n in range(1, N + 1)]] = np.nan
        self.combis["rho"] = pd.Series(None, dtype=object)

        reconstructed_rho = {
            n: {m: np.zeros((2, 2), dtype=complex) for m in range(1, N + 1)}
            for n in range(1, N + 1)
        }

        # For each Born decomposition...
        for c in self.combis.index:
            # generate the corresponding system...
            system = System(bomb_config, "0" + self.combis.at[c, "cleared"])
            system()

            # and save the probabilities.
            self.combis.at[c, range(1, N + 1)] = system.P.values

            # For each outcome...
            for outcome in range(1, N + 1):
                # and each bomb...
                for bomb in range(1, N + 1):
                    # construct the overall density matrix as per the Born
                    # decomposition
                    reconstructed_rho[outcome][bomb] += (
                        system.bombs[outcome][bomb][1:, 1:]
                        * self.combis.at[c, "weight"]
                        * self.combis.at[c, "prior"]
                    )
            self.combis.at[c, "rho"] = system.bombs

        # Check the Born decomposition as per Sinha et al.
        epsilon = self.combis[range(1, N + 1)].mul(
            self.combis["weight"] * self.combis["prior"], axis=0
        )
        assert np.allclose(epsilon.sum(axis=0).values, np.zeros(N))

        return reconstructed_rho

    @staticmethod
    def decompose_linearly(decomposed_rho):
        N = len(decomposed_rho["1"])

        weights = {
            outcome: {bomb: None for bomb in range(1, N + 1)}
            for outcome in range(1, N + 1)
        }

        for outcome in range(1, N + 1):
            for bomb in range(1, N + 1):
                # Decomposition of the bomb
                rho = {
                    k: decomposed_rho[k][outcome][bomb]
                    for k in decomposed_rho.keys()
                }

                # Pop and save the overall state with all paths cleared.
                rho_all = rho.pop("".join([str(j) for j in range(1, N + 1)]))

                # TODO: Try different combinations
                # Remove the single paths
                [rho.pop(str(x)) for x in list(range(1, N + 1))]

                vectors = {k: v.reshape(-1, 1) for k, v in rho.items()}

                # Stack the vectors horizontally
                vectors_keys = sorted(vectors.keys())

                # Return the least-squares solution.
                matrix = np.hstack([vectors[k] for k in vectors_keys])

                x, residuals, rank, s = np.linalg.lstsq(
                    matrix, rho_all.reshape(-1, 1), rcond=None
                )

                # from scipy import linalg
                # x, residuals, rank, s = linalg.lstsq(
                #     matrix, rho_all.reshape(-1, 1)
                # )

                # TODO: Round, and examine visually the decompositions. E.g.,
                # parity, etc.
                x = {
                    j: qi.trim_imaginary(x[i][0])
                    for i, j in enumerate(vectors_keys)
                }
                x["residuals"] = residuals

                weights[outcome][bomb] = x

        decomposition = {
            outcome: pd.DataFrame(weights[outcome])
            for outcome in weights.keys()
        }

        reconstruction_B_rho = {
            outcome: {
                bomb: np.sum(
                    [
                        decomposed_rho[k][outcome][bomb] * v
                        for k, v in decomposition[outcome][bomb][:-1].items()
                    ],
                    dtype=complex,
                    axis=0,
                )
                for bomb in range(1, N + 1)
            }
            for outcome in range(1, N + 1)
        }

        return decomposition, reconstruction_B_rho

    @staticmethod
    def compare_fidelities(rhoA, rhoB):
        for outcome in rhoA.keys():
            for bomb in rhoA[outcome].keys():
                print(
                    outcome,
                    bomb,
                    qi.fidelity(rhoA[outcome][bomb], rhoB[outcome][bomb]),
                )


# %% Run as a script, not as a module.
if __name__ == "__main__":
    bomb_config = "½½½½"  # ⅓t½ ⅓1½ ⅓t½½ ⅓t½⅓
    system = System(bomb_config, "0" + "1" * len(bomb_config))
    system()
    output_rho = system.bombs

    # TODO: Double-check manually and with the old numerical results

    # Perform the Born decomposition and save the reconstructed density
    # matrices.
    reconstruction_A_rho = system.decompose()

    # TODO: What did I "dream" about?

    # Save the density matrices generated from the Born decomposition.
    combis = system.combis
    decomposed_rho = combis.rho.to_dict()

    print("Probabilities:\n", np.round(system.P, ROUND), sep="")
    print(combis.drop("rho", inplace=False, axis=1))

    # Reconstruct the density matrices from the Born components.
    decomposition, reconstruction_B_rho = system.decompose_linearly(
        decomposed_rho
    )

    # TODO: Compute the fidelity between the reconstructions A and B, and the
    #       actual output density matrices.

    system.compare_fidelities(output_rho, reconstruction_A_rho)
    print("---")
    system.compare_fidelities(output_rho, reconstruction_B_rho)

# %%

# TODO: Check linear combination leading to rho for ABC

# df = pd.MultiIndex.from_product(
#     [range(1, 4), range(1, 4)], cleareds=["outcome", "bomb"]
# ).to_frame()

# for o in range(1, 4):
#     for b in range(1, 4):
#         print("outcome", o, b)
#         x, residuals = decompose_linearly(rho, o, b)
#         print(residuals)
#         print(x)
#         print()
#         # df.loc[o, b]['residuals'] = residuals
#         reconstr_bomb = sum([x[j] * rho[j][o][b] for j in x.keys()])
#         allclose = np.allclose(system.bombs[o][b], reconstr_bomb)


# %%
