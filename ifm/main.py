# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:04:59 2023

@author: Amine Laghaout
https://orcid.org/0000-0001-7891-4505

TODO
- Entanglement bi-partite or N-partite (cf. Hardy)
- Examine the plots
- Double-check all the math
- Dockerize, clean, etc.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantum_information as qi
import qutip as qt
import qutip.visualization as qtv

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = " ".join(
    [r"\usepackage{amsmath}", r"\usepackage{braket}"]
)


# %% System class


class System:
    def __init__(self, b, g=None):
        """
        Initialize the bomb and photon states.

        Parameters
        ----------
        b : TYPE
            DESCRIPTION.
        g : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        # Tuple of bomb states
        if isinstance(b, str):
            self.bomb_config = b
            self.b = tuple(
                # qi.BOMB_DICT[bomb]
                qi.BOMB_DICT.loc[bomb, "state"]
                for bomb in b
            )
        else:
            self.b = b
            assert False not in [isinstance(k, np.ndarray) for k in self.b]

        # Infer the number of paths form the number of bombs.
        self.N = len(self.b)

        # If unspecified, assume that the photon is in a uniform, coherent
        # superposition over all modes.
        if g is None:
            g = np.array([0] + [1] * self.N) / np.sqrt(self.N)
        # If specified…
        else:
            # as a string, convert it to a list of integers, and…
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

        # Initialize the post-photon-measurement states of the bombs keyed by
        # outcome (i.e., location of the detector click) and by bomb index.
        bombs = {
            outcome: {  # Outcome
                bomb: np.nan  # Bomb index (path location)
                * np.ones((2, 2), dtype=complex)
                for bomb in range(1, self.N + 1)
            }
            for outcome in range(1, self.N + 1)
        }

        # For each outcome…
        for outcome in range(1, self.N + 1):
            # retrieve the set of possible post-measurement state (i.e.,
            # basis vectors for the photonic state).
            kets = self.coeffs.index.to_list()

            # For each bomb…
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
                # basis ket. However, that 0th dimension can be ignored since
                # the photon reached the detector is therefore known to have
                # remained unexploded.
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

                # Assert that the current bomb state is a density matrix.
                bombs[outcome][b + 1] = qi.trim_imaginary(
                    bombs[outcome][b + 1], qi.TOL
                )
                assert qi.is_density_matrix(
                    bombs[outcome][b + 1]
                ), f"outcome {outcome}, bomb {b+1}:\n{bombs[outcome][b + 1]}\
                        \n{self.prob}"

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

        # There are 2ᴺ basis states.
        self.coeffs = pd.DataFrame(index=list(range(2**self.N)))

        # Convert to binary representation.
        self.coeffs["ket"] = self.coeffs.index.map(lambda x: bin(x)[2:])

        # Pad with zeros and add one to each "bit" so, e.g., 01 becomes 12,
        # meaning a photon on the first path and no photon on the second path.
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
        self.prob = self.coeffs.apply(
            lambda y: qi.trim_imaginary(np.conj(y) @ y, qi.TOL), axis=0
        )
        self.prob = pd.DataFrame(
            {"probability": self.prob.values}, index=self.prob.index
        )
        self.prob.index.rename("outcome", inplace=True)

        # Normalize the whole state.
        self.coeffs /= np.sqrt(self.prob["probability"])

    @staticmethod
    def Born_decomposition(N, k=2, delimiter=qi.DELIMITER, sort=True):
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

        modes = list(range(1, N + 1))
        P = {f"{k}": 0 for k in modes}
        for n in modes:
            P[f"{n}"] += 1 - N + n
            for m in range(n + 1, N + 1):
                P[f"{n}{delimiter}{m}"] = 1
                P[f"{m}"] -= 1

        df = pd.DataFrame.from_dict(P, orient="index", columns=["weight"])
        df.sort_index(inplace=True)

        # Binary representation of the cleared paths.
        df["cleared"] = df.index.map(
            lambda x: "".join(
                ["1" if str(j) in x else "0" for j in range(1, N + 1)]
            )
        )
        df["prior"] = df.index.map(lambda x: len(x.split(delimiter)) / N)

        return df

    def prep_report(self, bombs):
        rangeN = list(range(1, self.N + 1))
        index = pd.MultiIndex.from_product(
            [rangeN] * 2, names=["outcome", "bomb"]
        )

        columns = (
            [
                ("actual", "rho", "initial"),
                ("actual", "purity", "initial"),
                ("actual", "rho", "final"),
                ("actual", "purity", "final"),
            ]
            + [("actual", "rho", c) for c in self.combis.index]
            + [("actual", "purity", c) for c in self.combis.index]
            + [("Born", "rho", "final")]
        )
        columns = pd.MultiIndex.from_tuples(columns)

        df = pd.DataFrame(columns=columns, index=index)

        df[("actual", "rho", "final")] = df.apply(
            lambda x: bombs[x.name[0]][x.name[1]], axis=1
        )
        df[("actual", "purity", "final")] = df[
            ("actual", "rho", "final")
        ].apply(lambda x: qi.purity(x))
        df[("actual", "rho", "initial")] = df.apply(
            lambda x: np.outer(
                self.b[x.name[1] - 1][1:], np.conj(self.b[x.name[1] - 1][1:])
            ),
            axis=1,
        )
        df[("actual", "purity", "initial")] = df[
            ("actual", "rho", "initial")
        ].apply(lambda x: qi.purity(x))

        return df

    def get_rho(
        self,
        outcome,
        bomb,
        config="final",
        rho_type="actual",
        return_type="numpy",
    ):
        """
        Convert the Hilbert space to the Fock Hilbert space.

        Parameters
        ----------
        outcome : TYPE
            DESCRIPTION.
        bomb : TYPE
            DESCRIPTION.
        config : TYPE, optional
            DESCRIPTION. The default is "final".
        rho_type : TYPE, optional
            DESCRIPTION. The default is "actual".
        return_type : TYPE, optional
            DESCRIPTION. The default is "numpy".
         : TYPE
            DESCRIPTION.

        Returns
        -------
        rho : TYPE
            DESCRIPTION.
        """
        rho = self.report[rho_type].rho[config].loc[(outcome, bomb)].copy()
        rho = rho.T
        rho_00 = rho[0, 0]
        rho[0, 0] = rho[1, 1]
        rho[1, 1] = rho_00
        if return_type == "qutip":
            rho = qt.Qobj(rho)
        return rho

    def plot_report(self, resolution=200, optimize_pdf=True):
        N = self.N
        report = self.report

        fig, axes = plt.subplots(N, N + 1, figsize=(4 * N, 4 * N))

        # ### Diagonal in the Fock basis
        # Plot the diagonals of the Fock density matrix of each bomb. For each
        # such plot, mention the purity of the state.
        for b in range(1, N + 1):
            # Start with the initial state of the bomb.
            purity = qi.purity(report.actual.rho["initial"].loc[(1, b)])
            qtv.plot_fock_distribution(
                self.get_rho(1, b, "initial", return_type="qutip"),
                fig=fig,
                ax=axes[b - 1, 0],
            )

            # Only write the x-label for the lowest subplot (i.e., the last
            # bomb).
            if b == N:
                axes[b - 1, 0].set_xlabel("Fock diagonal")
            else:
                axes[b - 1, 0].set_xlabel(None)
            axes[b - 1, 0].set_ylabel("probability")
            rho_LaTeX = r"$\hat{\rho}^{(" + str(b) + ")}$ = "
            rho_LaTeX += qi.BOMB_DICT.loc[self.bomb_config[b - 1]].LaTeX
            axes[b - 1, 0].set_title(rho_LaTeX)
            axes[b - 1, 0].set_xticks(np.arange(-1, 3, 1))
            axes[b - 1, 0].set_xlim([-0.5, 1.5])

            # Then do the same for the states resulting from each detection
            # outcome.
            for o in range(1, N + 1):
                purity = qi.purity(report.actual.rho["final"].loc[(o, b)])
                qtv.plot_fock_distribution(
                    self.get_rho(o, b, "final", return_type="qutip"),
                    fig=fig,
                    ax=axes[b - 1, o],
                )

                # Only write the x-label for the lowest subplot (i.e., the last
                # bomb).
                if b == N:
                    axes[b - 1, o].set_xlabel("Fock diagonal")
                else:
                    axes[b - 1, o].set_xlabel(None)
                axes[b - 1, o].set_ylabel(None)
                rho_LaTeX = r"$\hat{\rho}^{(" + str(b) + ")}_{" + str(o) + "}$"
                axes[b - 1, o].set_title(
                    f"{rho_LaTeX}, purity = {np.round(purity, qi.ROUND)}"
                )
                axes[b - 1, o].set_xticks(np.arange(-1, 3, 1))
                axes[b - 1, o].set_xlim([-0.5, 1.5])

        fig.tight_layout()
        plt.savefig(f"Fock {self.bomb_config}.pdf")
        plt.show()

        # ### Wigner function
        # Do the same as the above, but now plot the Wigner functions instead.
        xvec = np.linspace(-2.7, 2.7, resolution)
        fig, axes = plt.subplots(N, N + 1, figsize=(4 * N, 4 * N))

        (vmin, vmax) = (-0.2, 0.32)
        # (vmin, vmax) = (-0.5, 0.5)

        cont = []
        lbl = []
        for b in range(1, N + 1):
            cont += [
                axes[b - 1, 0].contourf(
                    xvec,
                    xvec,
                    qt.wigner(
                        self.get_rho(1, b, "initial", return_type="qutip"),
                        xvec,
                        xvec,
                    ),
                    resolution,
                    vmin=vmin,
                    vmax=vmax,
                )
            ]
            rho_LaTeX = r"$\hat{\rho}^{(" + str(b) + ")}$ = "
            rho_LaTeX += qi.BOMB_DICT.loc[self.bomb_config[b - 1]].LaTeX
            lbl += [
                (
                    axes[b - 1, 0].set_title(rho_LaTeX),
                    axes[b - 1, 0].set_ylabel(r"$p$"),
                    axes[b - 1, 0].set_xlabel(r"$q$"),
                )
            ]
            if b == N:
                axes[b - 1, o].set_xlabel(r"$q$")
            # fig.colorbar(cont[-1], ax=lbl[-1], orientation='vertical')
            for o in range(1, N + 1):
                cont += [
                    axes[b - 1, o].contourf(
                        xvec,
                        xvec,
                        qt.wigner(
                            self.get_rho(o, b, "final", return_type="qutip"),
                            xvec,
                            xvec,
                        ),
                        resolution,
                        vmin=vmin,
                        vmax=vmax,
                        # cmap="viridis"
                        # norm='Normalize'
                    )
                ]
                rho_LaTeX = r"$\hat{\rho}^{(" + str(b) + ")}_{" + str(o) + "}$"
                lbl += [
                    (
                        axes[b - 1, o].set_title(rho_LaTeX),
                        axes[b - 1, o].set_ylabel(None),
                    )
                ]
                if b == N:
                    axes[b - 1, o].set_xlabel(r"$q$")
                # fig.colorbar(cont[-1], ax=lbl[-1], orientation='vertical')

        if optimize_pdf:
            for k in cont:
                for c in k.collections:
                    c.set_edgecolor("face")

        fig.tight_layout()
        plt.savefig(f"Wigner {self.bomb_config}.pdf")
        plt.show()

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
        for c in self.combis.index:
            # generate the corresponding system,
            subsystem = System(bomb_config, "0" + self.combis.at[c, "cleared"])
            subsystem()

            # check that its photon states are normalized,
            assert (
                np.round(qi.trim_imaginary(subsystem.g @ subsystem.g), qi.DEC)
                == 1
            )

            # save the outcome probabilities,
            self.combis.at[c, range(1, N + 1)] = subsystem.prob["probability"]

            # compute the density matrix of the post-photon-measurement state
            # of the bomb,
            self.report[("actual", "rho", c)] = self.report.apply(
                lambda x: subsystem.report.loc[
                    (x.name[0], x.name[1]),
                    ("actual", "rho", "final"),
                ],
                axis=1,
            )
            for x in self.report[("actual", "rho", c)]:
                assert qi.is_density_matrix(x)

            # along with its corresponding purity.
            self.report[("actual", "purity", c)] = self.report[
                ("actual", "rho", c)
            ].apply(qi.purity)

        # Check Born's rule as per Sinha et al.
        epsilon = (
            self.combis[range(1, N + 1)]
            .mul(self.combis["weight"] * self.combis["prior"], axis=0)
            .sum(axis=0)
        )
        assert np.allclose(epsilon.values, self.prob["probability"])

        # Check that the same decomposition also applies to the density
        # matrices.

        self.state_coeffs = []

        for outcome in self.report.index:
            state_coeffs = (
                self.combis.weight
                * self.combis.prior
                * self.combis[outcome[0]]
                / self.prob["probability"].loc[outcome[0]]
            )

            self.report[("Born", "rho", "final")].at[(outcome,)] = (
                self.report.actual.rho.loc[outcome, self.combis.index]
                @ state_coeffs
            )

            self.state_coeffs += [state_coeffs]

        self.state_coeffs = pd.concat(self.state_coeffs, axis=1).T
        self.state_coeffs.set_index(self.report.index, inplace=True)

        # Assert that the linear combination adds up to unity.
        assert (self.state_coeffs.sum(axis=1) - 1 < qi.TOL).all()

        ####
        linear_combinations = pd.DataFrame(
            self.state_coeffs.loc[
                [(k, 1) for k in range(1, self.N + 1)]
            ].values,
            index=range(1, self.N + 1),
            columns=self.combis.index,
        )
        linear_combinations.index.rename("outcome", inplace=True)
        linear_combinations["hash"] = linear_combinations.apply(
            lambda x: self.matrix2hash(x.values), axis=1
        )
        linear_combinations["sum"] = linear_combinations[
            self.combis.index
        ].sum(axis=1)
        self.prob = pd.concat([self.prob, linear_combinations], axis=1)
        ####

        self.report[("Born", "rho", "fidelity")] = self.report.apply(
            lambda x: qi.fidelity(
                x[("Born", "rho", "final")], x[("actual", "rho", "final")]
            ),
            axis=1,
        )

        def assertion(x, print_only=False, tol=qi.TOL):
            if print_only:
                print(
                    "Denisty matrix?",
                    qi.is_density_matrix(x[("Born", "rho", "final")]),
                )
                print(
                    "Fidelity?",
                    qi.fidelity(
                        x[("Born", "rho", "final")],
                        x[("actual", "rho", "final")],
                    ),
                )
            else:
                assert qi.is_density_matrix(x[("Born", "rho", "final")])
                assert (
                    abs(
                        qi.fidelity(
                            x[("Born", "rho", "final")],
                            x[("actual", "rho", "final")],
                        )
                        - 1
                    )
                    < tol
                )

        self.report.apply(lambda x: assertion(x, False, tol=qi.TOL), axis=1)

        self.hashed_rhos = self.matrices2hashes(
            self.report.actual.rho.map(self.matrix2hash),
            self.combis.index.to_list(),
        )
        self.hash_dict = self.hash2matrix(
            self.hashed_rhos,
            self.report.actual.rho,
            ["initial", "final"] + self.combis.index.to_list(),
        )

        # Check that all hashes are represented in the hash dictionary.
        assert set(
            self.hashed_rhos[
                ["initial", "final"] + self.combis.index.to_list()
            ]
            .values.flatten()
            .tolist()
        ) == set(self.hash_dict.keys())

        # Check that all head-on collisions are the same.
        assert self.hashed_rhos["H"].apply(lambda x: len(x) == 1).all()
        assert len(set(self.hashed_rhos["H"].apply(str))) == 1

        # Check that "intact" and "initial" are identical and unique per bomb.
        assert self.hashed_rhos["I"].apply(lambda x: len(x) == 1).all()
        assert (
            self.hashed_rhos[["initial", "I"]]
            .apply(lambda x: x["initial"] == x["I"][0], axis=1)
            .all()
        )

    @staticmethod
    def matrix2hash(matrix, encoding="utf-8", sha_round=qi.SHA_ROUND):
        import hashlib

        matrix = qi.trim_imaginary(matrix)

        matrix = str(np.round(matrix.reshape(-1, 1), sha_round))

        # Convert the string to bytes
        input_bytes = matrix.encode(encoding)

        # Create a sha256 hash object
        hash_object = hashlib.sha256(input_bytes)

        # Generate the hexadecimal representation of the SHA hash
        sha_value = hash_object.hexdigest()

        return sha_value[-sha_round:]

    @staticmethod
    def matrices2hashes(matrices, combis, sep=":"):
        matrices = matrices.copy()
        initial = {
            k[1]: f"i{k[0]}" for k in matrices.initial.loc[(1,)].items()
        }
        vertical = matrices.apply(lambda x: set(x), axis=0)
        horizontal = matrices.apply(lambda x: set(x), axis=1)

        matrices[combis] = matrices[combis].apply(
            lambda x: x + sep + x.index, axis=1
        )
        matrices["bomb"] = matrices.index.map(lambda x: str(x[1]))
        matrices[combis] = matrices.apply(
            lambda x: pd.Series([k + sep + x["bomb"] for k in x[combis]]),
            axis=1,
        )
        matrices.drop("bomb", axis=1, inplace=True)

        def prefixes(x, sep=sep, delimiter=qi.DELIMITER):
            x = tuple(x.split(sep))
            x = dict(
                matrix_hash=x[0],
                decomposition=x[1].split(delimiter),
                bomb=x[2],
            )

            # Head-on collision
            if (
                x["bomb"] in x["decomposition"]
                and len(x["decomposition"]) == 1
            ):
                x = "H" + sep + x["matrix_hash"]
            # Partial (IFM-like) interaction
            elif (
                x["bomb"] in x["decomposition"] and len(x["decomposition"]) > 1
            ):
                x = "P" + sep + x["matrix_hash"]
            # Inact (no photon incidence)
            elif x["bomb"] not in x["decomposition"]:
                x = "I" + sep + x["matrix_hash"]
            else:
                raise ValueError("Invalid post-measurement bomb status.")

            return x

        matrices[combis] = matrices[combis].map(prefixes)
        matrices["initial"] = matrices["initial"].map(lambda x: "I" + sep + x)
        matrices["final"] = matrices["final"].map(lambda x: "F" + sep + x)
        matrices["sets"] = matrices.apply(lambda x: set(x), axis=1)

        # Construct the sets
        for S in ["P", "H", "I"]:
            matrices[S] = matrices["sets"].map(
                lambda x: sorted([k for k in x if k.split(sep)[0] == S])
            )

        def check_one_element(x):
            assert len(x) == 1
            return x

        [matrices["H"].apply(check_one_element) for S in ["H", "I"]]

        matrices["size"] = matrices["sets"].apply(len)

        # matrices.drop(["sets", "H", "I"], axis=1, inplace=True)

        set_of_states = set(matrices[combis].values.flatten().tolist())
        for k in matrices.loc[1].index:
            set_of_states = set(
                matrices[combis].loc[k].values.flatten().tolist()
            )

        return matrices

    @staticmethod
    def hash2matrix(df1, df2, columns):
        # TODO: Check we're not overwriting duplicate keys in the dictionary
        hash_dict = dict(
            zip(df1[columns].values.flatten(), df2[columns].values.flatten())
        )
        return hash_dict

    def check_disturbance(
        self,
        P=None,
        U=None,
        coeffs=None,
        bomb_vectors=None,
        unitaries=dict(
            H=np.array([[1, 1], [1, -1]]) / np.sqrt(2), I=np.eye(2)
        ),
    ):
        if coeffs is None:
            coeffs = self.coeffs
        if bomb_vectors is None:
            bomb_vectors = self.b

        N = len(bomb_vectors)
        print(f"{U = }, {P = }")

        # Apply the unitary to all N bombs.
        if U is None:
            U = "H" * N
        U = qi.compose(tuple(unitaries[k] for k in list(U)))

        # Projective measurements on the bomb states after they've undergone
        # the unitary transformation. By default (P is None), the projective
        # measurement assumes a click on the first mode (which corresponds to a
        # Hadamard rotation of ½).
        if P is None:
            P = "0" * N
        # Otherwise, if P is provided as a string of 0s and 1s, the 0s
        # correspond  to a click on the first mode and the 1s to a click on the
        # second mode.  Normally, a Hadamard rotation of ½ state can only
        # possibly lead to a  measurement on the first mode. A click on the
        # second mode necessarily  means that the initial ½ state was disturbed
        # (presumably via the back-action of an interaction-free measurement).
        P = qi.compose(
            tuple(np.array([int(k == "0"), int(k == "1")]) for k in list(P))
        )

        # Construct the density matrices of the post-photon-measurement bombs.
        rho = {f"inter_{n}": qi.vec2mat(coeffs[n]) for n in range(1, N + 1)}

        # Include the density matrices of the intact bombs (i.e., prior to any
        # interaction with the photon. Start from the index 1 to avoid the
        # irrelevant first dimension of the Hilbert space corresponding to an
        # explosion. Note that the state vector is flipped so as to go from the
        # Hilbert space defined in the notes to the more traditional Fock
        # Hilbert space.
        rho["intact"] = qi.compose(tuple(j[1:][::-1] for j in bomb_vectors))

        # Validate the unitary, the projective measurement, and the states of
        # the bombs.
        assert qi.is_unitary(U) and qi.is_density_matrix(P)
        for k, r in rho.items():
            assert qi.is_density_matrix(r)

        print("Without photon interaction and back-action:")
        p = qi.Born(P, U @ rho["intact"] @ np.conjugate(U.T))
        print(f"p =", np.round(p, qi.ROUND))

        print("With photon interaction and back-action:")
        for n in range(1, N + 1):
            p = qi.Born(P, U @ rho[f"inter_{n}"] @ np.conjugate(U.T))
            print(f"p({n}) =", np.round(p, qi.ROUND))


# %% Run as a script, not as a module.
if __name__ == "__main__":
    """
    ½0 0½ 0000½ ½½½ ½½½½ ⅓0½ ⅓O½ ⅓O½½ ⅓O½⅓ ⅓O½⅓½ ½0 O0 O00 ⅓O½½ ⅓O½ ⅓vh½0h ⅓½½ ⅔½½
    ⅔0 ⅔00 ⅔000 0⅔ 00⅔ 000⅔ ⅔⅔ ⅔⅔⅔ ⅔⅔⅔⅔ ⅔⅔0 ⅔0⅔ 0⅔⅔ ⅔⅔00 ⅔0⅔0 0⅔0⅔ 0⅔⅔0 00⅔⅔
    ⅓vh½0⅔O½ ⅓vh½0⅔O ⅓vh½0⅔ ⅓vh½0 ⅓vh½ ⅓vh
    """
    systems = dict()
    for bomb_config in "½000 ½½00 ½½½0 ½½½½".split():  # ⅔h⅓ ⅓vh½0⅔
        systems[bomb_config] = System(
            bomb_config, "0" + "1" * len(bomb_config)
        )
        systems[bomb_config]()
        coeffs = systems[bomb_config].coeffs
        combis = systems[bomb_config].combis
        systems[bomb_config].decompose_born()
        prob = systems[bomb_config].prob
        print(f"{bomb_config}:")
        print(prob[["probability", "hash"]])
        report = systems[bomb_config].report
        hash_dict = systems[bomb_config].hash_dict
        hashed_rhos = systems[bomb_config].hashed_rhos
        state_coeffs = systems[bomb_config].state_coeffs
        # systems[bomb_config].plot_report(100, optimize_pdf=False)
        systems[bomb_config].check_disturbance(
            P="0" * systems[bomb_config].N,
            U="".join(["H" if k == "½" else "I" for k in list(bomb_config)]),
        )

        """
        U b -> P
        Undisturbed/inconclusive:
        H ½ -> 0
        I 0 -> 0
        """


# %%


def check_decomposition(rho_all, ROUND=qi.ROUND):
    N = len(rho_all)
    combis = System.Born_decomposition(N)
    index = combis.index
    total = 0

    # Normalize
    rho_all = rho_all / np.sqrt(rho_all @ np.conjugate(rho_all))
    rho_all_vector = rho_all.copy()
    print("*** rho_all_vector:")
    print(np.round(rho_all_vector, ROUND), "\n")

    # Density matrix from state vector
    rho_all = np.outer(rho_all, np.conjugate(rho_all))
    print("*** rho_all:")
    print(np.round(rho_all, ROUND), "\n")
    assert qi.is_density_matrix(rho_all)

    rho = {k: None for k in index}
    # For each decomposition
    for k in rho.keys():
        rho[k] = [0] * N
        for i in [int(j) - 1 for j in k.split(qi.DELIMITER)]:
            rho[k][i] = rho_all_vector[i]
        prior = qi.trim_imaginary(
            np.array(rho[k]) @ np.array(np.conjugate(rho[k]))
        )
        combis.loc[k, "prior"] = prior

        if prior != 0:
            # Normalize the state
            # rho[k] = rho[k] / np.sqrt(rho[k] @ np.conjugate(rho[k]))
            rho[k] = (
                np.outer(np.array(rho[k]), np.conjugate(np.array(rho[k])))
                * combis.loc[k]["weight"]
                # * combis.loc[k]["prior"]
            )
            total += rho[k]
        else:
            rho[k] = 0
    print("*** total:")
    print(np.round(total, ROUND), "\n")
    assert qi.is_density_matrix(total)
    assert np.allclose(total, rho_all)
    return dict(rho=rho, rho_all=rho_all, total=total, combis=combis)


# decomposition = check_decomposition(
#     np.array(
#         [0.5j, 0.5 + 1j, -1, 0, 0.65 - 3 * 1j, -3, 1]
#         # [1, 1 , .5, 1j]
#         # [1]*2,
#         # [1]* systems[bomb_config].N
#     )
# )

# %%
