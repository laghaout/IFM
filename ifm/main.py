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
DELIMITER = "·"

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
            # TODO: Change the range to go from 1 to N and replace b with b-1.
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
                assert qi.is_density_matrix(
                    bombs[outcome][b + 1]
                ), f"outcome {outcome}, bomb {b+1}:\n{bombs[outcome][b + 1]}\
                        \n{self.P}"

                # Move one "bit" to the right for the next bomb. Cf. the
                # manuscript for the full details.
                kets = [k[-1] + k[:-1] for k in kets]

        # TODO: Delete once we have the report
        self.bombs = bombs  # TODO: Delete when done.
        self.purity = pd.DataFrame({"prob": self.P})
        for k in range(1, self.N + 1):
            self.purity[k] = self.purity.apply(
                lambda x: qi.purity(self.bombs[x.name][k]), axis=1
            )

        self.report = self.prep_report(bombs)

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
    def Born_decomposition(N, k=2, delimiter=DELIMITER):
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
        comb_list = [delimiter.join([str(k) for k in range(1, N + 1)])]
        comb_list += S + [delimiter.join(sorted(c)) for c in comb]

        mydict = {f"{k}": 0 for k in comb_list}
        mydict[delimiter.join([str(k) for k in range(1, N + 1)])] = 1

        for n in range(1, N + 1):
            # print(f"P{n}")
            mydict[f"{n}"] -= 1
            for m in range(1, n):
                # print(f"P{m}{n}-P{n}-P{m}")
                mydict[f"{m}{delimiter}{n}"] -= 1
                mydict[f"{n}"] += 1
                mydict[f"{m}"] += 1

        df = pd.DataFrame(index=mydict.keys())
        df["cleared"] = df.index.map(
            lambda x: "".join(
                ["1" if str(j) in x else "0" for j in range(1, N + 1)]
            )
        )
        df["weight"] = df.index.map(lambda x: mydict[x])

        df["prior"] = df.index.map(lambda x: len(x.split(delimiter)) / N)

        return df

    def prep_report(self, bombs):
        components = self.Born_decomposition(self.N)
        self.combis = components
        components = components.index

        rangeN = list(range(1, self.N + 1))
        index = pd.MultiIndex.from_product(
            [rangeN] * 2, names=["outcome", "bomb"]
        )

        columns = (
            [
                (
                    "actual",
                    "rho",
                ),
                (
                    "actual",
                    "purity",
                ),
            ]
            + [("actual", "rho", c) for c in components[1:]]
            + [("actual", "purity", c) for c in components[1:]]
        )
        for reconstruction in ("born", "linear"):
            columns += (
                [
                    (
                        reconstruction,
                        "rho",
                    ),
                    (
                        reconstruction,
                        "purity",
                    ),
                    (
                        reconstruction,
                        "fidelity",
                    ),
                ]
                + [(reconstruction, "weight", c) for c in components[1:]]
                + [(reconstruction, "residuals")]
            )
        columns = pd.MultiIndex.from_tuples(columns)

        df = pd.DataFrame(columns=columns, index=index)

        df[("actual", "rho", None)] = df.apply(
            lambda x: bombs[x.name[0]][x.name[1]], axis=1
        )
        df[("actual", "purity", None)] = df[("actual", "rho", None)].apply(
            lambda x: qi.purity(x)
        )

        return df

    def decompose_born(self):
        # Retrieve the number of modes.
        N = self.N

        # Add the outcome probabilities and the density matrices for each
        # combination.
        self.combis[[n for n in range(1, N + 1)]] = np.nan
        self.combis["rho"] = pd.Series(None, dtype=object)

        # Prepare the reconstructed density matrix for each outcome and each
        # bomb.
        reconstructed_rho = {
            n: {m: np.zeros((2, 2), dtype=complex) for m in range(1, N + 1)}
            for n in range(1, N + 1)
        }

        # For each Born decomposition,
        for ci, c in enumerate(self.combis.index):
            # generate the corresponding system
            system = System(bomb_config, "0" + self.combis.at[c, "cleared"])
            system()

            # and save the outcome probabilities.
            self.combis.at[c, range(1, N + 1)] = system.P.values

            # For each outcome
            for outcome in range(1, N + 1):
                # and each bomb
                for bomb in range(1, N + 1):
                    # construct the overall density matrix as per the Born
                    # decomposition in Sinha et al. Note that we start at `ci`
                    # greater than zero so as to exclude the case where all
                    # modes are completely cleared. Note also that the factors
                    # are subtracted, not added, since they're already the
                    # negatives of what they should be. Cf. Eqs. (5) and (6) of
                    # Sinha et al.
                    if ci > 0:
                        reconstructed_rho[outcome][bomb] -= (
                            system.bombs[outcome][bomb]
                            * self.combis.at[c, "weight"]
                            * self.combis.at[c, "prior"]
                        )

            # TODO: Delete
            self.combis.at[c, "rho"] = system.bombs

        # Check the Born decomposition as per Sinha et al.
        epsilon = self.combis[range(1, N + 1)].mul(
            self.combis["weight"] * self.combis["prior"], axis=0
        )
        assert np.allclose(epsilon.sum(axis=0).values, np.zeros(N))

        # TODO: Make sure the reconstruction is a valid density matrix.
        for outcome in range(1, N + 1):
            for bomb in range(1, N + 1):
                # TODO: Replace by an assert statement.
                pass
                # assert qi.is_density_matrix(reconstructed_rho[outcome][bomb])
                # if not qi.is_density_matrix(
                #         reconstructed_rho[outcome][bomb]):
                #     print('>>', outcome, bomb)
                #     print(reconstructed_rho[outcome][bomb])

        for c in self.combis.index[1:]:
            self.report[("actual", "rho", c)] = self.report.apply(
                lambda x: self.combis.at[c, "rho"][x.name[0]][x.name[1]],
                axis=1,
            )

        return reconstructed_rho

    # @staticmethod
    def decompose_linearly(self, delimiter=DELIMITER):
        decomposed_rho = self.combis.rho.to_dict()
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
                rho_all = rho.pop(
                    delimiter.join([str(j) for j in range(1, N + 1)])
                )

                # TODO: Try different combinations
                # Pop one more?
                # TODO: Remove False
                if N > 3:
                    thelist = list(range(1, N + 1))
                    # [rho.pop(str(x)) for x in thelist]
                    # rho.pop('12')

                # The density matrices of the undisturbed paths are redundant.
                # Remove all but one by setting them to zero.
                cleared_paths = [k for k in rho.keys() if str(bomb) not in k]
                if len(cleared_paths) > 0:
                    for k in cleared_paths[:-1]:
                        rho[k] = np.zeros((2, 2))

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

                if len(cleared_paths) > 0:
                    for k in cleared_paths[:-1]:
                        x[k] = 0

                weights[outcome][bomb] = x

        decomposition_linear = {
            outcome: pd.DataFrame(weights[outcome])
            for outcome in weights.keys()
        }

        reconstruction_linear = {
            outcome: {
                bomb: np.sum(
                    [
                        decomposed_rho[k][outcome][bomb] * v
                        for k, v in decomposition_linear[outcome][bomb][
                            :-1
                        ].items()
                    ],
                    dtype=complex,
                    axis=0,
                )
                for bomb in range(1, N + 1)
            }
            for outcome in range(1, N + 1)
        }

        for outcome in range(1, N + 1):
            for bomb in range(1, N + 1):
                assert qi.is_density_matrix(
                    reconstruction_linear[outcome][bomb]
                ), f"outcome {outcome}, bomb {bomb}:\n{reconstruction_linear[outcome][bomb]}"

                # if not qi.is_density_matrix(reconstruction_linear[outcome][bomb]):
                #     print(outcome, bomb)

        return decomposition_linear, reconstruction_linear


# %% Run as a script, not as a module.
if __name__ == "__main__":
    # Try all ½½½ ½½½½ ⅓t½ ⅓1½ ⅓t½½ ⅓t½⅓ ⅓t½⅓½ ½0 10 100
    bomb_config = "⅓t½"
    system = System(bomb_config, "0" + "1" * len(bomb_config))
    system()
    report = system.report
    # print(report.actual)
    dtypes = report.dtypes
    actual = report.actual
    bombs = system.bombs
    reconstructed_Born = system.decompose_born()
    combis = system.combis
    decomposition_linear, reconstruction_linear = system.decompose_linearly()
    epsilon = system.combis[range(1, system.N + 1)].mul(
        system.combis["weight"] * system.combis["prior"], axis=0
    )

# %%
if False:

    def lexsort_error_demo():
        import pandas as pd

        N = 3
        rangeN = list(range(1, N + 1))
        index = pd.MultiIndex.from_product(
            [rangeN, rangeN], names=["level1", "level2"]
        )
        columns = [
            (
                "col1",
                "col1.1",
            ),
            (
                "col1",
                "col1.2",
            ),
        ]
        components = range(1, 3)
        columns += [("Col2", "Col2.1", f"Col2.1.{c}") for c in components]
        columns += [("Col2", "Col2.2", f"Col2.2.{c}") for c in components]
        columns = pd.MultiIndex.from_tuples(columns)
        # print(columns.is_lexsorted())

        df = pd.DataFrame(columns=columns, index=index).sort_index(axis=1)
        # df = df.sort_index(axis=1)

        df.loc[
            :,
            (
                "col1",
                "col1.2",
            ),
        ] = 7

        print(df)
