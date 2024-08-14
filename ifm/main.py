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


class System(BaseModel):
    qubits: tuple
    # Determined from the qubits
    photon: np.ndarray = None
    state: np.ndarray = None
    N: int = None
    report: pd.DataFrame = None
    jaja: None = None  # TODO: remove

    class Config:
        arbitrary_types_allowed = True

    def __call__(self):
        # Qubits state
        self.N = len(self.qubits)
        qubits = tuple(qi.pure_qubit(**qubit) for qubit in self.qubits)
        self.qubits = qubits[0]
        if self.N > 1:
            for q in range(1, self.N):
                self.qubits = sp.tensorproduct(self.qubits, qubits[q])
            self.qubits = sp.Matrix(self.qubits.reshape(2**self.N))

        # Photon state
        N = sp.symbols("N")
        self.photon = sp.ones(self.N, 1) / sp.sqrt(N)

        # Photon-qubits state
        self.state = sp.tensorproduct(self.photon, self.qubits)
        self.state = sp.Matrix(self.state.reshape(
            len(self.photon) * len(self.qubits), 1))

        # self.interact()
        self.beamsplit()
        self.measure_photon()
        # self.measure_qubits()

    def interact(self):
        print("==== Photon-qubit interaction")

    def beamsplit(self):
        print("==== Beam splitter transformation")

        BS = sp.tensorproduct(qi.symmetric_BS(self.N), sp.eye(2**self.N))        
        BS = BS.reshape(len(self.state), len(self.state))
        BS = sp.Matrix(BS)
        if self.state.shape[1] == 1:
            self.state = BS * self.state
        elif self.state.shape[0] == self.state.shape[1]:
            self.state = BS * self.state * BS.H

    def measure_photon(self):
        print("==== Measure the photon")
        P = []
        # For each possible position of a photon click...
        for n in range(1, self.N + 1):
            projector = sp.zeros(self.N)
            projector[n - 1, n - 1] = 1
            projector = sp.kronecker_product(projector, sp.eye(2**self.N))
            probability = qi.Born_rule(projector, self.state)
            P.append(probability)

        self.report = pd.DataFrame({"probability": P})
        self.report["probability_evaluated"] = self.report.probability.apply(
            lambda x: x.subs(dict(N=self.N)).evalf()
        )

        print(self.report.probability_evaluated)

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
        # qubits=(dict(q0=f"a{k}", q1=f"b{k}") for k in range(1, 4)),
        qubits=(dict(q0=k) for k in '½½')
    )
    system()
    report = system.report
    # %%
    
    # try:
    #     print(report.evaluated.apply(round))
    # except BaseException:
    #     print(report.evaluated)

