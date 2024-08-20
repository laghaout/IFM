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
from sympy.physics.quantum import TensorProduct
from types import SimpleNamespace


class System(BaseModel):
    qubits: tuple
    # Determined from the qubits
    photon: np.ndarray = None
    state: np.ndarray = None
    N: int = None
    report: pd.DataFrame = None
    BS: None = None  # TODO: remove
    x: dict = SimpleNamespace(**dict())

    class Config:
        arbitrary_types_allowed = True

    def __call__(self):  # CHECKED
        # Pure qubits state vector
        self.N = len(self.qubits)
        qubits = tuple(qi.pure_qubit(**qubit) for qubit in self.qubits)
        self.qubits = qubits[0]
        if self.N > 1:
            for q in range(1, self.N):
                self.qubits = TensorProduct(self.qubits, qubits[q])

        # Pure photon state vector
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
        # self.measure_qubits()

    def interact(self):
        print("==== Photon-qubit interaction")

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
            ].apply(lambda x: float(x.subs(dict(N=self.N)).evalf()))
        except BaseException:
            pass
        print(report.T)

    def measure_qubits(self):
        print("==== Measure the qubits")
        for n in range(1, self.N + 1):
            print(f"- Photon measurement at {n}:")
            for n in range(1, self.N + 1):
                print(f"  - Qubit measurement {n}:")


if __name__ == "__main__":
    system = System(
        # qubits=tuple(dict(q0=k) for k in '½'*3),
        # qubits=(dict(q0='a', q1='b'), dict(q0='x', q1='y'),),
        qubits=(dict(q0=f"a{k}", q1=f"b{k}") for k in range(1, 3)),
        # qubits=(dict(q0=k) for k in '½½')
    )
    system()
    system.disp_report()
    # %%

    # try:
    #     print(report.evaluated.apply(round))
    # except BaseException:
    #     print(report.evaluated)

# %%
A = tuple(bin(j)[2:] for j in range(8))
A = tuple("0" * (len(A[-1]) - len(j)) + j for j in A)
A = np.array(A)
print(A)
