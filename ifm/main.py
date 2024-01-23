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
import seaborn as sns

ROUND = 4
np.set_printoptions(precision=ROUND, suppress=True)

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


class IFM:
    def __init__(self, bombs, gamma=1, tol=qi.TOL, validate=True):
        """
        Parameters
        ----------
        bombs : tuple of np.ndarray or str
            Quantum states of the bombs. These can be specified explicitly in a
            tuple of NumPy arrays or as a string of characters where each
            character encodes a coherent superposition of the bomb being on the
            path of the photon and away from it. E.g., `⅓` represents the case
            where the photons is coherently ⅓ on the path of the photon and ⅔
            away from it.
            Each bomb lives in a 3-dimensional Hilbert space where
            - dimension 0 represents the exploded bomb,
            - dimension 1 represents the bomb in the photon's path, and
            - dimension 2 represents the bomb away from the photon's path.
            There are `N` bombs and all of them are in pure states.
        gamma : np.ndarray or int, optional
            Quantum state of the incident photon. It can be specified
            explicitly in Hilbert space as as nd.ndarray or implicitly as an
            integer specifying which path is excited by the photon (with all
            other paths assumed to be empty.)
            The photon lives in a 3-dimensional Hilbert space where
            - dimension 0 means that the photon was absorbed by a bomb,
            - dimension `k` means that the photon travels in path `k` of the
              `N`-armed Mach-Zehnder interferometer.
        tol : float, optional
            Numerical tolerance.
        validate : bool
            Validate the math (e.g., unitarity of the beam splitter, )?
        bomb_dict : dict
            Dictionary which maps the shorthand character for a bomb's state to
            a vector representation in the bomb Hilbert space.

        Returns
        -------
        None.
        """

        # If the bombs are represented by a string of characters and the
        # photon by the path it is exciting, generate the corresponding state
        # vectors.
        if isinstance(bombs, str) and isinstance(gamma, int):
            self.setup = f"{gamma}›{bombs}"  # Record the initial setup
            bombs = tuple(bomb_dict[b.upper()] for b in bombs)
            gamma_temp = np.zeros(len(bombs) + 1)
            gamma_temp[gamma] = 1
            gamma = gamma_temp
        elif isinstance(bombs, str) and isinstance(gamma, np.array):
            self.setup = f"-›{bombs}"  # Record the initial setup
            bombs = tuple(bomb_dict[b.upper()] for b in bombs)
            assert gamma.shape[0] == len(bombs) + 1
        else:
            self.setup = "-›-"

        self.validate = validate  # Validation flag
        self.tol = tol  # Numerical tolerance
        self.gamma = gamma  # Photonic state over the modes
        self.dims = [self.gamma.shape[0]]  # Dimensions of the Hilbert spaces
        self.bombs = bombs  # Bomb states
        self.modes = len(self.gamma) - 1  # Number of photon paths

        # Construct a symmetric `N`-mode beam splitter. Note that this beam
        # splitter has one extra dimension to account for the fact that the
        # incident vacuum remains unchanged.
        self.BS = qi.symmetric_BS(self.modes)
        if self.validate:
            assert self.modes == self.BS.shape[0] - 1
        if self.validate:
            assert qi.is_unitary(self.BS)

        # Construct the overall bomb state vector.
        self.dims += [bomb.shape[0] for bomb in self.bombs]
        self.bombs = bombs[0]
        for k in range(self.modes - 1):
            self.bombs = np.kron(self.bombs, bombs[k + 1])

        # Consrtuct the initial density matrix.
        if len(self.gamma.shape) == 1:
            input_state = np.kron(self.gamma, self.bombs)
            self.rho = np.outer(input_state, input_state)
        elif len(self.gamma.shape) == 2 and self.gamma.shape[1] > 1:
            self.rho = np.kron(self.gamma, np.outer(self.bombs, self.bombs))
        else:
            assert False

        if self.validate:
            assert qi.is_density_matrix(self.rho)

    def __call__(self, interact=True, verbose=True):
        """
        Run the photon through the Mach-Zehnder interferometer.

        Parameters
        ----------
        interact : bool, optional
            Shall the photon interact with the bombs?
        verbose : bool, optional
            Print updates on the evolution of the system?

        Returns
        -------
        None.
        """

        # Results
        self.results = {
            m: {
                "measurement": None,  # Measurement operator
                "probability": None,  # Outcome probability
                "post_rho": None,  # Overall output state
                "subsystems": {  # Output subsystems
                    "subrho": None,  # Output subsystem states
                    "purity": None,  # Purity of subsystem state
                    "diagonals": None,  # Diagonal of subsystem states
                },
            }
            for m in range(self.modes + 1)
        }

        # %% Initial density matrix

        if verbose:
            print(f"\n== {self.setup} ================== Initial state:")
            self.compute_substates(self.rho, self.modes, self.dims)

        # %% The first beam splitter

        # Generalize the beam splitter operation so that that it can act on the
        # photon-bombs system.
        BS = np.kron(self.BS, np.eye(self.bombs.shape[0]))
        if self.validate:
            assert qi.is_unitary(BS)

        # Operate the beam splitter on the state.
        self.rho = qi.BS_op(BS, self.rho)
        if self.validate:
            assert qi.is_density_matrix(self.rho)

        if verbose:
            print(f"\n== {self.setup} ======== After the first beam splitter")
            self.compute_substates(self.rho, self.modes, self.dims)

        # %% The photon-bombs interactions

        if interact:
            # TODO: Check that the order of the interactions does not matter.
            self.interac()
            if self.validate:
                assert qi.is_density_matrix(self.rho)

            if verbose:
                print(
                    f"\n== {self.setup} ======== After the photon-bombs interactions"
                )
                self.compute_substates(self.rho, self.modes, self.dims)

        # %% The second beam splitter

        self.rho = qi.BS_op(BS, self.rho)
        if self.validate:
            assert qi.is_density_matrix(self.rho)

        if verbose:
            print(f"\n== {self.setup} ======== After the second beam splitter")
            self.compute_substates(self.rho, self.modes, self.dims)

        # %% Photon measurements in the Fock basis

        if verbose:
            print(f"\n== {self.setup} ======== After the photon measurements")

        # For each measurement output…
        for k in self.results.keys():
            # Measurement operator

            # The projection operators as state vectors are all initialized to
            # 0. There are as many projection operators as the dimension of the
            # Hilbert space, which---recall---also include a projection on the
            # vacuum, corresponding to the absorption of the photon upon the
            # explosion of the bomb.
            measurement = np.zeros(self.modes + 1)
            # Select the dimension of the Hilbert space to be projected on.
            measurement[k] = 1
            # Construct the corresponding measurement operator.
            measurement = np.outer(measurement, measurement)
            # The Hilbert space of the bombs is not projected upon, so just
            # expand the measurement operator with the identity.
            measurement = np.kron(measurement, np.eye(len(self.bombs)))
            # Check the Hermicity of the measurement operator thus constructed.
            if self.validate:
                assert qi.is_Hermitian(measurement)
            self.results[k]["measurement"] = measurement

            # Probability of the measurement

            probability = qi.Born(measurement, self.rho)
            self.results[k]["probability"] = probability

            # Post-measurement state

            if probability > self.tol:
                post_rho = measurement @ self.rho @ measurement / probability
                if self.validate:
                    assert qi.is_density_matrix(post_rho)
                self.results[k]["post_rho"] = qi.trim_imaginary(post_rho)
            else:
                post_rho = None

            if verbose:
                print(
                    f"\n== {self.setup} === Prob({k}): {np.round(probability, ROUND)}"
                )
            self.results[k]["subsystems"] = self.compute_substates(
                post_rho, self.modes, self.dims, verbose=verbose
            )

        # Ensure the probabilities add up to unity.
        probabilities = [
            self.results[k]["probability"] for k in range(self.modes + 1)
        ]
        assert -self.tol <= sum(probabilities) <= 1 + self.tol

    def interac(self):  # TODO: Check this function
        rho = self.rho.copy()

        for mode in range(1, self.modes + 1):
            # Transitions
            starting_states = self.interac_helper(mode, False)
            ending_states = self.interac_helper(mode, True)
            state_transitions = list(zip(starting_states, ending_states))
            # print(f'Interaction at mode {mode}:')
            # print(state_transitions)

            C = np.eye(self.rho.shape[0])

            for trans in state_transitions:
                S = np.zeros(self.rho.shape)
                S[trans[0], trans[0]] = 1
                P = np.eye(self.rho.shape[0])
                row = P[trans[0], :].copy()
                P[trans[0], :] = P[trans[1], :]
                P[trans[1], :] = row

                C[trans[0], :] = 0

                rho += (
                    P @ S @ rho
                    + rho @ S.conj().T @ P.conj().T
                    + P @ S @ rho @ S.conj().T @ P.conj().T
                )

                rho = C @ rho @ C.conj().T

            self.rho = rho

    def interac_helper(self, mode, status):
        # TODO: What is going on here?
        if status is False:
            state = np.array([0, 1, 0])
        else:
            state = np.array([1, 0, 0])

        bomb = np.array([1, 1, 1])
        bombs = bomb
        if mode == 1:
            bombs = state
        for m in range(1, self.modes):
            if m == mode - 1:
                bombs = np.kron(bombs, state)
            else:
                bombs = np.kron(bombs, bomb)

        gamma = np.zeros(self.modes + 1)
        if status is False:
            gamma[mode] = 1
        else:
            gamma[0] = 1

        return np.where(np.kron(gamma, bombs) == 1)[0]

    def compute_substates(self, rho, modes, dims, verbose=True):
        subsystems = dict()

        if rho is None:
            if verbose:
                print("Impossible scenario.")
            return None
        else:
            subrho = qi.trim_imaginary(qi.partial_trace(rho, [0], dims))
            if self.validate:
                assert qi.is_density_matrix(subrho)
            purity = qi.purity(subrho, self.tol)
            if verbose:
                print("photon, purity:", np.round(purity, ROUND))
                print(np.round(subrho, ROUND))
            subsystems["photon"] = {
                "diagonals": subrho.diagonal(),
                "purity": purity,
                "subrho": subrho,
            }
            for k in range(modes):
                subrho = qi.trim_imaginary(
                    qi.partial_trace(rho, [k + 1], dims)
                )
                if self.validate:
                    assert qi.is_density_matrix(subrho)
                purity = qi.purity(subrho, self.tol)
                if verbose:
                    print(f"bomb {k+1}, purity:", np.round(purity, ROUND))
                    print(np.round(subrho, ROUND))
                subsystems[f"bomb {k+1}"] = {
                    "diagonals": subrho.diagonal(),
                    "purity": purity,
                    "subrho": np.round(subrho, qi.DEC),
                }

        return subsystems

    @staticmethod
    def plot_probabilities(
        values, title=None, xlabel="Mode", ylabel="Probability"
    ):
        # TODO: Move to quantum_information.py
        # TODO: Check all values are actually probabilities, i.e., real,
        #       between 0 and 1, and adding up to unity.

        # Create a Seaborn histogram
        sns.set(
            style="whitegrid",
            rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15},
        )
        # plt.figure(figsize=(10, 6))  # Set the figure size (adjust as needed)

        # Create the histogram using Seaborn
        xs = list(range(len(values)))
        xs[0] = "XPL"
        sns.barplot(
            x=xs,
            y=values,
            palette=["darkred"] + ["darkblue"] * (len(values) - 1),
        )

        # Set labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # Show the plot
        # plt.xticks()  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Ensure all elements are visible
        plt.show()

    def report(self):
        # Plot the probabilities of the different outcomes.
        probabilities = [
            self.results[m]["probability"] for m in range(self.modes + 1)
        ]
        self.plot_probabilities(
            probabilities, f"{self.setup}: measurement probabilities"
        )

        # For each measurement outcome (i.e., photon clicks)...
        for m in range(self.modes + 1):
            # If the outcome is possible...
            if self.results[m]["subsystems"] is not None:
                # Determine whether the outcome is a click or an explosion.
                if m != 0:
                    outcome = f"click at {m}"
                else:
                    outcome = "explosion"
                # For each bomb, plot the diagonals and the purity
                # TODO: also plot the fidelity wih the initial state.
                for b in range(1, self.modes + 1):
                    purity = np.round(
                        self.results[m]["subsystems"][f"bomb {b}"]["purity"],
                        ROUND,
                    )
                    self.plot_probabilities(
                        qi.trim_imaginary(
                            self.results[m]["subsystems"][f"bomb {b}"][
                                "diagonals"
                            ],
                        ),
                        f"{self.setup}: {outcome}, bomb {b}, "
                        + f"purity = {purity}",
                        xlabel="Diagonal",
                    )
            else:
                print(f"Skipping outcome {m} for setup `{self.setup}`")


# %% Extra functions


def closed_form(g, b):
    b_str = b
    b = np.vstack((np.array([np.nan] * 3),) + tuple(bomb_dict[x] for x in b))

    results = {
        m: {"probability": None, "subsystems": None}
        for m in range(1, len(b_str) + 1)
    }

    for m in results.keys():
        pm = (-1) ** (m + 1)

        P = (
            np.abs(g[1] * b[1, 2] * b[2, 1]) ** 2
            + np.abs((g[1] + pm * g[2]) * b[1, 2] * b[2, 2]) ** 2
            + np.abs(g[2] * b[1, 1] * b[2, 2]) ** 2
        ) / 2

        bomb = {
            "bomb 1": np.array(
                [
                    [
                        np.abs(g[2] * b[1, 1] * b[2, 2]) ** 2,
                        g[2]
                        * b[1, 1]
                        * b[2, 2]
                        * np.conj((g[1] + pm * g[2]) * b[1, 2] * b[2, 2]),
                    ],
                    [
                        pm
                        * (g[1] + pm * g[2])
                        * b[1, 2]
                        * b[2, 2]
                        * np.conj(g[2] * b[1, 1] * b[2, 2]),
                        np.abs(g[1] * b[1, 2] * b[2, 1]) ** 2
                        + np.abs((g[1] + pm * g[2]) * b[1, 2] * b[2, 2]) ** 2,
                    ],
                ]
            )
            / (2 * P)
            if P > 0
            else None,
            "bomb 2": np.array(
                [
                    [
                        np.abs(g[1] * b[1, 2] * b[2, 1]) ** 2,
                        g[1]
                        * b[1, 2]
                        * b[2, 1]
                        * np.conj((g[1] + pm * g[2]) * b[1, 2] * b[2, 2]),
                    ],
                    [
                        (g[1] + pm * g[2])
                        * b[1, 2]
                        * b[2, 2]
                        * np.conj(g[1] * b[1, 2] * b[2, 1]),
                        np.abs((g[1] + pm * g[2]) * b[1, 2] * b[2, 2]) ** 2
                        + np.abs(g[2] * b[1, 1] * b[2, 2]) ** 2,
                    ],
                ]
            )
            / (2 * P)
            if P > 0
            else None,
        }

        results[m]["probability"] = P
        results[m]["subsystems"] = bomb

    print("\nChecking the closed-form expression...")
    for m in results.keys():
        print(
            f"== ›{b_str} === Prob({m}):",
            np.round(results[m]["probability"], ROUND),
        )
        for k in results[m]["subsystems"].keys():
            print(
                f"{k}, purity:",
                np.round(qi.purity(results[m]["subsystems"][k]), ROUND),
            )
            print(results[m]["subsystems"][k])

    return results


def check_reconstruction(
    results, initial_bomb, base_config, my_config, outcome, bomb, k
):
    undisturbed = np.outer(bomb_dict[initial_bomb], bomb_dict[initial_bomb])
    disturbed = qi.get_subrho(base_config, outcome, bomb, results)
    final = qi.get_subrho(my_config, outcome, bomb, results)

    reconstructed_disturbed = qi.reconstruct_disturbed(
        N=len(my_config), final=final, undisturbed=undisturbed, k=k
    )

    if np.allclose(reconstructed_disturbed, disturbed, qi.TOL) is True:
        print("Good!")
    else:
        print("=============== Bad!")
        print(disturbed)
        print(reconstructed_disturbed)


def decompose(config, outcome, bomb, verbose=False):
    base = config[int(bomb.split(" ")[1]) - 1]

    Y = qi.get_subrho(config, outcome, bomb, results, None)[1:, 1:].reshape(
        -1, 1
    )

    # Convex components
    R = dict(
        undisturbed=np.outer(bomb_dict[base], bomb_dict[base])[1:, 1:].reshape(
            -1, 1
        ),
        # c=np.array([[1,0], [0,0]]).reshape(-1, 1),
        # r=np.array([[0,0], [0,1]]).reshape(-1, 1),
        # m=np.array([[bomb_dict[base][1]**2,0], [0, bomb_dict[base][2]**2]]).reshape(-1, 1),
        equally_collapsed=np.array([[1, 0], [0, 1]]).reshape(-1, 1) / 2,
    )
    R_keys = sorted(R.keys())

    # Return the least-squares solution.
    components = np.hstack([R[m] for m in R_keys])
    x, residuals, rank, s = np.linalg.lstsq(components, Y, rcond=None)

    assert len(residuals) > 0

    # Convex coefficients
    x = {j: qi.trim_imaginary(x[i][0]) for i, j in enumerate(R_keys)}

    # Assemble all the elements of the linear equation.
    E = {m: v.reshape(2, 2) for m, v in R.items()} | dict(
        final=Y.reshape(2, 2)
    )

    reconstructed_final = np.sum([v * E[i] for i, v in x.items()], axis=0)

    if verbose:
        print(f"Results of the least-squares fit for {config}:")
        print("- Residuals:", residuals)
        print("- Coefficients:")
        for k in R_keys:
            print(f"  - {k} =", np.round(x[k], ROUND))

        if not np.allclose(reconstructed_final, E["final"], qi.TOL):
            print("Mismatch!")
            print("Reconstructed:")
            print(reconstructed_final)
            print("Actual:")
            print(E["final"])
        else:
            print("Matching!")

    return (x, residuals, E, reconstructed_final)


# %% Run as a script, not as a module.
if __name__ == "__main__":
    # my_system = {k: None for k in
    #              ['⅔0',     '⅔00',  '⅔000', '⅔0000',
    #               '⅔⅔',     '⅔⅔0',  '⅔⅔00', '⅔⅔000',
    #               '⅔⅔⅔',    '⅔⅔⅔0', '⅔⅔⅔00',
    #               '⅔⅔⅔⅔',   '⅔⅔⅔⅔0',
    #               '⅔⅔⅔⅔⅔',
    #               '½0',     '½00',  '½000', '½0000',
    #               '½½',     '½½0',  '½½00', '½½000',
    #               '½½½',    '½½½0', '½½½00',
    #               '½½½½',   '½½½½0',
    #               '½½½½½',
    #               '⅓0',     '⅓00',  '⅓000', '⅓0000',
    #               '⅓⅓',     '⅓⅓0',  '⅓⅓00', '⅓⅓000',
    #               '⅓⅓⅓',    '⅓⅓⅓0', '⅓⅓⅓00',
    #               '⅓⅓⅓⅓',   '⅓⅓⅓⅓0',
    #               '⅓⅓⅓⅓⅓'
    #              ]}
    # my_system = "½00"  # ½⅓⅔
    my_system = None
    # my_system = False

    if isinstance(my_system, dict):
        results = my_system.copy()
        for k in my_system.keys():
            my_system[k] = IFM(k)
            my_system[k]()
            results[k] = my_system[k].results
            my_system[k].report()

        df = qi.results_vs_N(results, outcome=2, subsystem="bomb 1")
        # print(df)
        # check_reconstruction(
        #     results=results,
        #     initial_bomb="⅓",
        #     base_config="⅓⅓",
        #     my_config="⅓⅓⅓⅓",
        #     outcome=2,
        #     bomb="bomb 1",
        #     k=2,
        # )
    elif isinstance(my_system, str):
        my_bombs = my_system
        my_system = IFM(my_system)
        my_system()
        results = my_system.results
        my_system.report()

        # Check the closed-form results matches the numercial results.
        if len(my_bombs) == 2:
            results_closed_form = closed_form(
                np.array([0, 1, 1]) / np.sqrt(2), my_bombs
            )
            for m in sorted(results_closed_form.keys()):
                assert np.allclose(
                    results[m]["probability"],
                    results_closed_form[m]["probability"],
                    qi.TOL,
                )
                for k in results_closed_form[m]["subsystems"].keys():
                    assert np.allclose(
                        results[m]["subsystems"][k]["subrho"][1:, 1:],
                        results_closed_form[m]["subsystems"][k],
                        qi.TOL,
                    )
            print("Closed-form expression checked!")

    elif my_system is None:
        results = qi.reload()
        print(qi.get_subrho("½½00", 2, "bomb 1", results))
        df = qi.results_vs_N(results, outcome=2, subsystem="bomb 1")
        f, residuals, E, reconstructed_final = decompose(
            "⅔⅔⅔0", 2, "bomb 1", True
        )

    elif my_system is False:
        print("Pass")

# %%


class System:
    def __init__(self, b, g=None):
        self.b = tuple(bomb_dict[bomb] for bomb in b)
        self.N = len(self.b)

        # If unspecified, assume the photon is in a uniform coherent
        # superposition over all modes.
        if g is None:
            self.g = np.array([0] + [1] * self.N) / np.sqrt(self.N)

        self.BS = qi.symmetric_BS(self.N, include_vacuum_mode=True)

    def compute_coeffs(self):
        """
        Compute the coefficients of the basis states when there is no
        explostion.
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

        return self.coeffs / self.P

    def compute_coeff(self, n, z):
        Z = np.prod([self.b[m][int(x)] for m, x in enumerate(z)])
        Z *= np.multiply(self.BS[1:, n], self.g[1:]) @ np.array(
            [j == "2" for j in z]
        )

        return Z


# %%

config = "⅔⅔0"
curr_bomb = 1
outcome = 2

S = System(config)
Z_df = S.compute_coeffs()

A = Z_df.index.to_list()
N = len(Z_df.index[0])
rho_bomb = {j: np.zeros((2, 2), dtype=complex) for j in range(1, S.N + 1)}
for j in range(S.N):
    # Start-end pairs
    # bomb = [(A[k], A[k + 2 ** (N - 1)],) for k in range(1, 2 ** (N - 1))]
    bomb = [
        (
            Z_df.loc[A[k]][outcome],
            Z_df.loc[A[k + 2 ** (N - 1)]][outcome],
        )
        for k in range(1, 2 ** (N - 1))
    ]
    # Alone term
    # bomb += [A[2 ** (N - 1)]]
    bomb += [Z_df.loc[A[2 ** (N - 1)]][outcome]]

    # Move one "bit" to the right for the next bomb
    A = [a[-1] + a[:-1] for a in A]
    rho_bomb[j + 1][0, 0] = np.sum(
        [bomb[m][0] * np.conj(bomb[m][0]) for m in range(len(bomb) - 1)]
    )
    rho_bomb[j + 1][0, 1] = np.sum(
        [bomb[m][0] * np.conj(bomb[m][1]) for m in range(len(bomb) - 1)]
    )
    rho_bomb[j + 1][1, 0] = np.sum(
        [bomb[m][1] * np.conj(bomb[m][0]) for m in range(len(bomb) - 1)]
    )
    rho_bomb[j + 1][1, 1] = np.sum(
        [bomb[m][1] * np.conj(bomb[m][1]) for m in range(len(bomb) - 1)]
        + [bomb[len(bomb) - 1] * np.conj(bomb[len(bomb) - 1])]
    )
    rho_bomb[j + 1] *= S.P[outcome]
    rho_bomb[j + 1] = qi.trim_imaginary(rho_bomb[j + 1], qi.TOL)

print(rho_bomb[curr_bomb])
print(qi.get_subrho(config, outcome, f"bomb {curr_bomb}", results)[1:, 1:])

print(
    np.allclose(
        qi.get_subrho(config, outcome, f"bomb {curr_bomb}", results)[1:, 1:],
        rho_bomb[curr_bomb],
    )
)
