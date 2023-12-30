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
            There are `N` bombs.
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
        else:
            # If the initial setup hasn't been recorded as a string, just set
            # it to None.
            self.setup = None

        # Ensure the number of bombs is equal to the number of photon paths.
        if validate:
            assert gamma.shape[0] == len(bombs) + 1
        for bomb in bombs:
            if validate:
                assert isinstance(bomb, np.ndarray)

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

        # Consrtuct the initial density matrix.
        input_state = np.kron(self.gamma, self.bombs)
        self.rho = np.outer(input_state, input_state)
        if self.validate:
            assert qi.is_density_matrix(self.rho)

        # results
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


# %%


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


# %% Run as a script, not as a module.
if __name__ == "__main__":
    # my_system = {k: None for k in
    #              ['⅔0',     '⅔00',  '⅔000',
    #               '⅔⅔',     '⅔⅔0',  '⅔⅔00',
    #               '⅔⅔⅔',    '⅔⅔⅔0', '⅔⅔⅔⅔',
    #               '½0',     '½00',  '½000',
    #               '½½',     '½½0',  '½½00',
    #               '½½½', '½½½0', '½½½½',
    #               '⅓0', '⅓00', '⅓000',
    #               '⅓⅓', '⅓⅓0', '⅓⅓00',
    #               '⅓⅓⅓', '⅓⅓⅓0', '⅓⅓⅓⅓'
    #              ]}
    my_system = "⅔000"  # ½⅓⅔

    if isinstance(my_system, dict):
        results = my_system.copy()
        for k in my_system.keys():
            my_system[k] = IFM(k)
            my_system[k]()
            results[k] = my_system[k].results
            my_system[k].report()

        df = qi.results_vs_N(results, outcome=2, subsystem="bomb 1")
        print(df)
        check_reconstruction(
            results=results,
            initial_bomb="⅓",
            base_config="⅓⅓",
            my_config="⅓⅓⅓⅓",
            outcome=2,
            bomb="bomb 1",
            k=2,
        )
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

    # %% Closed form expression

    # results_closed_form = closed_form(
    #     np.array([0, 1, 1]) / np.sqrt(2), '⅔⅔')  # ½⅓⅔
    # # A = results_closed_form[2]['subsystems']['bomb 1']
    # # B = results_closed_form[2]['subsystems']['bomb 2']

    # %% Reloaded results

    import quantum_information as qi

    results = qi.reload()
    print(qi.get_subrho("½½00", 2, "bomb 1", results))

# %%

import numpy as np

# Define the matrix B and the vector v
B = np.array([[0.5, 1, 0.5], [0.5, 0, 0], [0.5, 0, 0], [0.5, 0, 0.5]])
v = np.array([0.75, 0.25 * (1 - 1j), 0.25 * (1 + 1j), 0.25])

k = 0.2357022603955
B = np.array(
    [
        [1 / 3, 1, 0.5],
        [np.sqrt(2) / 3, 0, 0],
        [np.sqrt(2) / 3, 0, 0],
        [2 / 3, 0, 0.5],
    ]
)
v = np.array([2 / 3, k * (1 - 1j), k * (1 + 1j), 1 / 3])

# Solve for x
# x = np.linalg.solve(B, v)
x, residuals, rank, s = np.linalg.lstsq(B, v, rcond=None)

print("Residual:", residuals)
print(x)
