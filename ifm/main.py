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

        # For each outcome,
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

        self.combis = self.Born_decomposition(self.N)

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
        # clearedly a photon on the first path and no photon on the second
        # path.
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
    def Born_decomposition(N, k=2, delimiter=DELIMITER, sort=True):
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

        # Convert iterator to a list
        comb_list = [delimiter.join([str(k) for k in range(1, N + 1)])]
        comb_list += S + [delimiter.join(sorted(c)) for c in comb]

        # Sort the Born components alphabetically to avoid `lexsort` warnings
        # later on.
        comb_list = [comb_list[0]] + sorted(comb_list[1:])

        # Weights of the various components. These weights correspond to the
        # signs in front of each term in Eq. (6) of Sinha et al.
        weights = {f"{k}": 0 for k in comb_list}
        weights[delimiter.join([str(k) for k in range(1, N + 1)])] = 1

        for n in range(1, N + 1):
            weights[f"{n}"] -= 1
            for m in range(1, n):
                weights[f"{m}{delimiter}{n}"] -= 1
                weights[f"{n}"] += 1
                weights[f"{m}"] += 1

        df = pd.DataFrame(index=weights.keys())

        # Binary representation of the cleared paths.
        df["cleared"] = df.index.map(
            lambda x: "".join(
                ["1" if str(j) in x else "0" for j in range(1, N + 1)]
            )
        )
        df["weight"] = df.index.map(lambda x: weights[x])
        df["prior"] = df.index.map(lambda x: len(x.split(delimiter)) / N)

        # Add the outcome probabilities and the density matrices for each
        # combination.
        # df[[n for n in range(1, N + 1)]] = np.nan
        # df["rho"] = pd.Series(None, dtype=object)

        return df

    def prep_report(self, bombs):
        components = self.combis.index

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
                            system.report.loc[
                                (outcome, bomb), ("actual", "rho", None)
                            ]
                            * self.combis.at[c, "weight"]
                            * self.combis.at[c, "prior"]
                        )
                    # assert qi.is_density_matrix(reconstructed_rho[outcome][bomb])

            # compute the density matrix of the pre-measurement state,
            self.report[("actual", "rho", c)] = self.report.apply(
                lambda x: system.report.loc[
                    (x.name[0], x.name[1]), ("actual", "rho", None)
                ],
                axis=1,
            )
            # its corresponding purity,
            self.report[("actual", "purity", c)] = self.report.apply(
                lambda x: qi.purity(
                    system.report.loc[
                        (x.name[0], x.name[1]), ("actual", "rho", None)
                    ]
                ),
                axis=1,
            )
            # as well as its coefficient. See Eq. (6) of Sinha et al.
            self.report[("born", "weight", c)] = (
                self.combis.at[c, "weight"] * self.combis.at[c, "prior"]
            )

        # Check the decomposed Born probabilities as per Sinha et al.
        epsilon = self.combis[range(1, N + 1)].mul(
            self.combis["weight"] * self.combis["prior"], axis=0
        )
        assert np.allclose(epsilon.sum(axis=0).values, np.zeros(N))

        # TODO: Use a dot product instead of the nested loop above
        self.report[("born", "rho", None)] = self.report.apply(
            lambda x: reconstructed_rho[x.name[0]][x.name[1]],
            axis=1,
        )

        self.report[("born", "purity", None)] = self.report[
            ("born", "rho", None)
        ].apply(lambda x: qi.purity(x) if qi.is_density_matrix(x) else np.nan)
        self.report[("born", "fidelity", None)] = self.report.apply(
            lambda x: qi.fidelity(
                x[("born", "rho", None)], x[("actual", "rho", None)]
            )
            if qi.is_density_matrix(x[("born", "rho", None)])
            else np.nan,
            axis=1,
        )

    # TODO: Consider doing the linear regression over all the bombs and
    # outcomes at once, and not just per bomb-outcome.
    def decompose_linear(self, delimiter=DELIMITER):
        # Reshape the matrices as vectors
        matrix = self.report.actual.rho.map(lambda x: x.reshape(-1, 1))
        decompositions = matrix.columns[1:]

        # TODO: Drop some terms
        # if self.N > 3:
        #     decompositions = matrix.columns[1:]
        # else:
        #     decompositions = matrix.columns[1:]

        # self.zouz = matrix.copy()
        # print(decompositions)
        # print(matrix)

        self.report.loc[:, ("linear", "weight")] = 0
        matrix["Vecs"] = matrix.apply(
            lambda x: np.hstack([x[k] for k in decompositions]), axis=1
        )
        matrix.drop(decompositions, axis=1, inplace=True)
        # matrix['res'] = matrix.apply(lambda x: x[np.nan], axis=1)
        # x[np.nan] represents the actual rho
        matrix["res"] = matrix.apply(
            lambda x: np.linalg.lstsq(
                x["Vecs"], x[np.nan].reshape(-1, 1), rcond=None
            ),
            axis=1,
        )
        self.matrix = matrix
        for i, col in enumerate(decompositions):
            self.report.loc[:, ("linear", "weight", col)] = matrix[
                "res"
            ].apply(
                lambda x: qi.trim_imaginary(x[0][i][0])
            )  # x[0][c]

        self.report.loc[:, ("linear", "residuals", None)] = matrix[
            "res"
        ].apply(lambda x: x[0][1])

        self.report.loc[:, ("linear", "rho", slice(None))] = self.report.apply(
            lambda x: (
                x[("actual", "rho")][decompositions]
                @ x[("linear", "weight")][decompositions]
            ),
            axis=1,
        )

        self.report.loc[:, ("linear", "rho", slice(None))] = self.report.loc[
            :, ("linear", "rho", slice(None))
        ].apply(lambda x: x.values.item(), axis=1)

        self.report[("linear", "purity", None)] = self.report[
            ("linear", "rho", None)
        ].apply(lambda x: qi.purity(x) if qi.is_density_matrix(x) else np.nan)
        self.report[("linear", "fidelity", None)] = self.report.apply(
            lambda x: qi.fidelity(
                x[("actual", "rho", None)], x[("linear", "rho", None)]
            )
            if qi.is_density_matrix(x[("linear", "rho", None)])
            else np.nan,
            axis=1,
        )


# %% Run as a script, not as a module.
if __name__ == "__main__":
    # Try all ½½½ ½½½½ ⅓t½ ⅓1½ ⅓t½½ ⅓t½⅓ ⅓t½⅓½ ½0 10 100
    bomb_config = "⅓t½½"  # ⅓t½½ ⅓t½
    system = System(bomb_config, "0" + "1" * len(bomb_config))
    system()
    system.decompose_born()
    system.decompose_linear()
    report = system.report
    matrix = system.matrix
    print(report.linear.fidelity)

# %% Temporary

if True:

    def check_with_old(
        report=report,
        reconstruction_linear=reconstruction_linear,
        decomposed_rho=decomposed_rho,
    ):
        for o, b in report.index:
            print(
                np.allclose(
                    report.loc[(o, b), ("linear", "rho")].values.item(),
                    reconstruction_linear[o][b],
                ),
                "linear reconstruction",
            )

            print(
                np.allclose(
                    report.loc[(o, b), ("actual", "rho", None)],
                    output_rho[o][b],
                ),
                "actual",
            )

            decomp = "1" + DELIMITER + "3"
            print(
                np.allclose(
                    report.loc[(o, b), ("actual", "rho", decomp)],
                    decomposed_rho[decomp][o][b],
                ),
                "decomposed_rho",
            )
            # print(
            #     (
            #         report.loc[(o, b), ("linear", "rho")].values.item()
            #         - reconstruction_linear[o][b]
            #     ).sum()
            # )
            # print(
            #     (
            #         report.loc[(o, b), ("linear", "rho")].values.item()
            #         - report.loc[(o, b), ("linear", "rho")].values.item()
            #     ).sum()
            # )

    check_with_old()
