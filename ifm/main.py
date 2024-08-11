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
    # Other
    report: pd.DataFrame = None

    class Config:
        arbitrary_types_allowed = True

    def __call__(self):
        # Assemble the qubit state vector.
        self.N = len(self.qubits)
        qubits = tuple(qi.pure_qubit(**qubit) for qubit in self.qubits)
        self.qubits = qubits[0]
        if self.N > 1:
            for q in range(1, self.N):
                self.qubits = sp.tensorproduct(self.qubits, qubits[q])
            self.qubits = sp.Matrix(self.qubits.reshape(2**self.N))

        # Assemble the photon state vector.
        N = sp.symbols("N")
        self.photon = sp.ones(self.N, 1) / sp.sqrt(N)

        # Assemble the overall state vector.
        self.state = sp.tensorproduct(self.photon, self.qubits)
        self.state = sp.Matrix(
            self.state.reshape(len(system.photon) * len(system.qubits))
        )

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
        self.state = sp.Matrix(BS) @ sp.Matrix(self.state)

    def measure_photon(self):
        print("==== Measure the photon")
        P = []
        for n in range(1, self.N + 1):
            projector = np.zeros(self.N)
            projector[n - 1] = 1
            projector = np.kron(
                np.outer(projector, projector), np.eye(2**self.N, dtype=int)
            )
            A = qi.Born(projector, self.state)
            P.append(qi.Born(projector, self.state))

        self.report = pd.DataFrame({"prob": P})
        self.report["evaluated"] = self.report.prob.apply(
            lambda x: x.subs(dict(N=self.N)).evalf()
        )

        print(self.report.evaluated)

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
        qubits=(dict(q0=f"a{k}", q1=f"b{k}") for k in range(1, 4)),
        # qubits=(dict(q0=k) for k in '½½½')
    )
    system()
    # %%
    # report = system.report
    # try:
    #     print(report.evaluated.apply(round))
    # except BaseException:
    #     print(report.evaluated)


# %%
