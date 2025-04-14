# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:58:27 2024

@author: amine
"""

import pandas as pd
from pydantic import BaseModel
import numpy as np
import quantum_information as qi
import sympy as sp
from sympy.physics.quantum import Ket, TensorProduct
from types import SimpleNamespace


class System(BaseModel):
    qubits: tuple
    # Determined from the qubits
    photon: np.ndarray = None
    realign: np.ndarray = None
    state: np.ndarray = None
    N: int = None
    report: pd.DataFrame = None
    BS: None = None  # TODO: remove
    x: dict = SimpleNamespace(**dict())
    skip: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __call__(self):  # CHECKED
        # Pure qubits state vector
        self.N = len(self.qubits)
        qubits = tuple(qi.pure_qubit(**qubit) for qubit in self.qubits)
        self.qubits = qubits[0]
        self.realign = qi.align_with_0(qubits[0])

        if self.N > 1:
            for q in range(1, self.N):
                self.realign = TensorProduct(
                    self.realign, qi.align_with_0(qubits[q])
                )
                self.qubits = TensorProduct(self.qubits, qubits[q])

        # Pure photon state vector that is equally delocalized over the N modes
        N = sp.symbols("N")
        self.photon = sp.ones(self.N, 1) / sp.sqrt(N)

        # Pure photon-qubits state vector
        self.state = TensorProduct(self.photon, self.qubits)

        # Generate the report such that the photon-click outcomes are the
        # columns. The rows consist of the
        # - probability for the photon-click outcome, and
        # - the post-photon-click state of the qubits.
        index = (("probability", None),) + tuple(
            ("qubit", int(k)) for k in range(1, self.N + 1)
        )
        self.report = pd.DataFrame(
            columns=range(1, self.N + 1),
            index=pd.MultiIndex.from_tuples(index),
        )

        self.interact()
        self.beamsplit()
        self.measure_photon()
        self.measure_qubits()

    def interact(self):
        print("==== Photon-qubit interaction")

        # Define the Hilbert space.
        photon = list(range(1, self.N + 1))
        qubit = [0, 1]
        Hilbert = SimpleNamespace(
            **dict(
                bases=[photon] + [qubit] * self.N,
                names=["photon"]
                + [f"qubit_{k}" for k in range(1, self.N + 1)],
            )
        )

        # Basis of the Hilbert space
        index = pd.MultiIndex.from_product(Hilbert.bases, names=Hilbert.names)
        b = {n: sp.symbols(f"{n}") for n in range(0, self.N + 1)}
        basis = pd.DataFrame(
            dict(
                # Coefficients of each basis vector. Note that the 0-photon
                # subspace corresponding to the "exploded" qubit are
                # initialized to 0.
                coeff=list(self.state),
                # Basis vector in ket notation
                eigenvector=index.map(lambda x: Ket(*[b[j] for j in x])),
                # Interaction on the basis vector? If so, specify which qubit
                # is involved. (Zero means no qubit was involved.)
                interaction=index.map(lambda x: (x[x[0]] != 0) * x[0]),
                # interaction=index.map(lambda x: x[x[0]] != 0),
            ),
            index=index,
        )
        basis["interaction"] = basis["interaction"].map(
            lambda x: False if x == 0 else x
        )
        
        
        if self.skip is False: 
            # Interacted subspace
            self.x.interacted = basis[basis["interaction"] != False]

            # Post-selected non-interacted state: Set the interacted coefficients
            # to zero.
            self.state = basis["coeff"].copy()
            self.state[basis["interaction"] != 0] = 0
            self.state = sp.Matrix(self.state.values)

        self.x.basis = basis

    def beamsplit(self):  # CHECKED
        print("==== Beam splitter transformation")

        # Apply the beam splitter transformation to the photon only.
        BS = TensorProduct(qi.symmetric_BS(self.N), sp.eye(2**self.N))
        BS = BS.reshape(len(self.state), len(self.state))
        BS = sp.Matrix(BS)
        self.BS = BS
        assert qi.is_unitary(self.BS)

        # Transform the state depending on whether it is a vector or a matrix.
        if self.state.shape[1] == 1:
            self.state = self.BS * self.state
        elif self.state.shape[0] == self.state.shape[1]:
            self.state = self.BS * self.state * BS.H

    def measure_photon(self):  # CHECKED
        print("==== Measure the photon")

        # For each possible position of the photon click,
        for n in range(1, self.N + 1):
            # assemble the projector over the whole state and measure the
            # probability of projection.
            projector = sp.zeros(self.N)
            projector[n - 1, n - 1] = 1
            projector = sp.kronecker_product(projector, sp.eye(2**self.N))
            probability = qi.Born_rule(projector, self.state)
            # Evaluate numerically if possible.
            self.report.loc[("probability", None)][n] = probability

    def disp_report(self):
        report = self.report.T
        try:
            report[("probability", None)] = report[
                ("probability", None)
            ].apply(
                lambda x: qi.round(
                    np.complex64(x.subs(dict(N=self.N)).evalf())
                )
            )
        except BaseException:
            pass
        print(report.T)
        
        print(
            "|state| =",
            (self.state.H * self.state).subs(dict(N=self.N)).evalf()[0],
        )
        if self.skip is False:
            print(
                "|interacted| =",
                (self.x.interacted.coeff @ np.conjugate(self.x.interacted.coeff))
                .subs(dict(N=self.N))
                .evalf(),
            )
        print(
            "|basis| =",
            (self.x.basis.coeff @ self.x.basis.coeff.map(np.conj))
            .subs(dict(N=self.N))
            .evalf(),
        )
    
        print(self.x.df)        

    def post_photon_click(self, photon):
        mask = self.x.basis.index.get_level_values("photon") == photon

        # Use numpy to find the integer positions where the mask is True
        integer_indices = list(np.where(mask)[0])

        return self.state[list(integer_indices), :]

    def measure_qubits(self):
        print("==== Check the alignment of the qubits")

        def helper(alignments, click_at):
            # Select the state after the particular photon click.
            state = self.post_photon_click(click_at)

            # zaza = state.subs(dict(N=self.N)).evalf()
            # # state = sp.Matrix([[.], [.5], [.5], [0]])
            # print(zaza)

            # Try to realign all the qubits.
            A = qi.unitary_transform(self.realign, state)

            # Joint measurement to check the alignment of all qubits.
            B = qi.Born_rule(qi.measurement_on_energy(alignments), A)

            probability = B
            
            try: 
                probability = B.subs(dict(N=self.N)).evalf()
                probability = qi.sympy_round(probability)
            except BaseException:
                pass

            return probability

        index = [bin(j)[2:] for j in range(2**self.N)]
        index = ["0" * (len(index[-1]) - len(j)) + j for j in index]
        df = pd.DataFrame(columns=range(1, self.N + 1), index=index)

        for j in df.columns:
            df[j] = df.index.map(lambda x: helper(x, j))

        self.x.df = df

# %%

if __name__ == "__main__":
    # qubits = tuple(dict(q0=k) for k in '½'*2)
    # qubits = (dict(q0='a', q1='b'), dict(q0='x', q1='y'),)
    # qubits = (dict(q0=f"a{k}", q1=f"b{k}") for k in range(1, 3))
    # qubits = (dict(q0=k) for k in "⅓0i") # ½⅓⅔ij01
    qubits = (dict(q0=k) for k in "½½")  # ½⅓⅔ij01  j⅔⅓  j⅔zx
    qubits = tuple(qubits)
    system = System(
        qubits=qubits,  
        # skip=True
    )
    system()
    system.disp_report()
    x = system.x
    state = system.state
    # print(x.df)

#%%
if False:
    N = 3
    a = np.sqrt(1/2)
    b = np.sqrt(2/3)
    c = np.sqrt(3/4)
    d = np.sqrt(6/7)
    
    
    # P_000_1 = ((a**2 + b**2 + c**2)/N)**2
    # P_100_1 = (a*np.sqrt(1-a**2)/N)**2
    # P_001_1 = (c*np.sqrt(1-c**2)/N)**2
    # P_010_1 = (b*np.sqrt(1-b**2)/N)**2
    # print(f"{P_000_1 = }")
    # print(f"{P_100_1 = }")
    # print(f"{P_010_1 = }")
    # print(f"{P_001_1 = }")
    
    P_000_2 = ((a**2 + b**2 + c**2 + d**2)/N)**2
    P_100_2 = (a*np.sqrt(1-a**2)/N)**2
    P_001_2 = (c*np.sqrt(1-c**2)/N)**2
    P_010_2 = (b*np.sqrt(1-b**2)/N)**2
    print(f"{P_000_2 = }")
    print(f"{P_100_2 = }")
    print(f"{P_010_2 = }")
    print(f"{P_001_2 = }")
    
