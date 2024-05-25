# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:04:59 2023

@author: Amine Laghaout

Generalization of the Elitzur-Vaidman bomb tester to `N`-partite Mach-Zehnder
interferometers where the bombs can be in a superposition of being on the path
and away from the path.
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
        self.prob = pd.Series(
            self.prob.values, name="probability", index=self.prob.index
        )

        # Normalize the whole state.
        self.coeffs /= np.sqrt(self.prob)

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
            self.combis.at[c, range(1, N + 1)] = subsystem.prob

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
        assert np.allclose(epsilon.values, self.prob)

        # Check that the same decomposition also applies to the density
        # matrices.
        # decompositions = self.report.actual.rho[self.combis.index]
        # weights = self.combis["weight"] * self.combis["prior"]
        # self.report[("Born", "rho", "reconstructed")] = decompositions @ weights

        for outcome in self.report.index:
            self.report[("Born", "rho", "final")].at[
                (outcome,)
            ] = self.report.actual.rho.loc[outcome, self.combis.index] @ (
                self.combis.weight
                * self.combis.prior
                * self.combis[outcome[0]]
                / self.prob[outcome[0]]
            )

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

        self.report.apply(lambda x: assertion(x, False, tol=1e-8), axis=1)


# %% Run as a script, not as a module.
if __name__ == "__main__":
    """
    ½½½ ½½½½ ⅓0½ ⅓O½ ⅓O½½ ⅓O½⅓ ⅓O½⅓½ ½0 O0 O00 ⅓O½½ ⅓O½ ⅓vh½0h ⅓½½ ⅔½½
    ⅔0 ⅔00 ⅔000 0⅔ 00⅔ 000⅔ ⅔⅔ ⅔⅔⅔ ⅔⅔⅔⅔ ⅔⅔0 ⅔0⅔ 0⅔⅔ ⅔⅔00 ⅔0⅔0 0⅔0⅔ 0⅔⅔0 00⅔⅔
    """
    systems = dict()
    for (
        bomb_config
    ) in "½0 0½ 0000½ ½½½ ½½½½ ⅓0½ ⅓O½ ⅓O½½ ⅓O½⅓ ⅓O½⅓½ ½0 O0 O00 ⅓O½½ ⅓O½ ⅓vh½0h ⅓½½ ⅔½½".split():
        systems[bomb_config] = System(
            bomb_config, "0" + "1" * len(bomb_config)
        )
        systems[bomb_config]()
        coeffs = systems[bomb_config].coeffs
        combis = systems[bomb_config].combis
        prob = systems[bomb_config].prob
        print(f"{bomb_config}:")
        print(prob)
        systems[bomb_config].decompose_born()
        report = systems[bomb_config].report
        # systems[bomb_config].plot_report(100, optimize_pdf=False)

# %%

# for out in report.index:
#     A = report.actual.rho.loc[out, combis.index] @ (combis.weight * combis.prior * combis[out[0]] / prob[out[0]])
#     print(f"{out} = {qi.fidelity(A, report.actual.rho.loc[out, 'final'])} ({qi.is_density_matrix(A)})")
