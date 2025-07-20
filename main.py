# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:52:40 2025

@author: Amine Laghaout
"""

from functools import reduce
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional


class IFM(BaseModel):
    psi_photon: list                                # Initial photonic state
    psi_qubits: list                                # Initial qubit states
    psi: Optional[object] = None                    # Overall state
    N: int = None                                   # Number of qubits
    sep: str = '⊗'                                  # Qubit-photon separator
    interaction: str | None = 'Elitzur-Vaidman'     # Photon-qubit interaction
    outcomes: object = None                         # Summary of outcomes
    steps: set = ('interact', 'beam_split', 'realign', 'measure')

    def __call__(self):

        # By default, if the number of qubits is specified, use an equal
        # distribution of the photonic and qubit modes.
        if isinstance(self.N, int):
            self.psi_photon = np.hstack([np.zeros(1), np.ones(self.N)])
            self.psi_qubits = np.ones([self.N, 2])
        else:
            self.psi_photon = np.array(self.psi_photon, dtype=complex)
            self.psi_qubits = np.array(self.psi_qubits, dtype=complex)
            self.N = len(self.psi_qubits)

        self.validate()

        # Assemble the overall state.
        psi_qubits = reduce(np.kron, self.psi_qubits)
        self.psi = np.kron(psi_qubits, self.psi_photon)
        index_tuples = [(k, n) for k in range(2**(self.N))
                        for n in range(self.N + 1)]
        index = pd.MultiIndex.from_tuples(index_tuples, names=["k", "n"])
        self.psi = pd.DataFrame(index=index, data={"amplitude": self.psi})
        self.psi['ket'] = self.psi.index.map(
            lambda k: self.int_to_binary(k[0], self.N)+self.sep+str(k[1]))

        if 'interact' in self.steps:
            self.interact()     # The photon and qubit states interact.
        if 'beam_split' in self.steps:
            self.beam_split()   # The photon undergoes a beam splitting.
        if 'realign' in self.steps:
            self.realign()      # The qubits are realigned.
        if 'measure' in self.steps:
            self.measure()      # Measure.

    def validate(self):

        assert self.N == len(self.psi_photon) - 1 == self.psi_qubits.shape[0]
        assert self.psi_qubits.shape[1] == 2

        # Normalize the photonic state.
        norm_photon = np.sqrt(self.psi_photon @ np.conjugate(self.psi_photon))
        if not np.isclose(norm_photon, 1):
            self.psi_photon = self.psi_photon / norm_photon
            norm_photon = np.sqrt(
                self.psi_photon @ np.conjugate(self.psi_photon))
            assert np.isclose(norm_photon, 1)
            # print("Normalizing the photonic state.")

        # Normalize the qubit states.
        norm_qubits = self.psi_qubits * np.conjugate(self.psi_qubits)
        norm_qubits = np.sqrt(norm_qubits.sum(axis=1)).reshape(-1, 1)
        if not np.all(
                np.isclose(norm_qubits, np.ones(self.psi_qubits.shape[0]))):
            self.psi_qubits = self.psi_qubits / norm_qubits
            norm_qubits = self.psi_qubits * np.conjugate(self.psi_qubits)
            norm_qubits = np.sqrt(norm_qubits.sum(axis=1))
            assert np.all(
                np.isclose(norm_qubits, np.ones(self.psi_qubits.shape[0])))
            # print("Normalizing the qubits states.")

    def interact(self):

        # print(f"==== Interact ({self.interaction})")

        match self.interaction:
            case "Elitzur-Vaidman":
                # Identify the modes where the photon undergoes a "head-on
                # collision" with the qubit in its path. This will determine
                # which 0-photon (i.e., absorbed photon) state will acquire
                # the probability amplitude.
                self.psi['collision'] = self.psi.apply(
                    lambda x: (x.name[0], 0) if x.ket[x.name[1]-1] == '1'
                    else None,
                    axis=1)
                collided = self.psi[self.psi['collision'].notna()].copy()
                for index, row in collided.iterrows():
                    self.psi.loc[
                        row['collision'], 'amplitude'] += row['amplitude']
                    self.psi.loc[index, 'amplitude'] = 0
            case _:
                pass

        norm = self.norm(self.psi.amplitude)
        # print("Norm before normalization:", norm)
        self.psi.amplitude = self.psi.amplitude / norm
        norm = self.norm(self.psi.amplitude)
        # print("Norm after normalization:", norm)

    def beam_split(self):
        # print("==== Beam-split")
        self.psi.amplitude = self.symmetric_BS(self.N) @ self.psi.amplitude

    def measure_at(self, n: int = None, qubits: str = None) -> float:
        mask = np.ones(len(self.psi))

        if n is not None:       # Measure the photon.
            mask *= self.psi.index.map(lambda x: 1 if x[1] == n else 0)
        if qubits is not None:  # Measure the qubits.
            mask *= self.psi.ket.apply(
                lambda x: 1 if x.split(self.sep)[0] == qubits else 0)

        amplitude = mask * self.psi.amplitude
        probability = amplitude @ np.conjugate(amplitude)

        return self.iround(probability)

    def measure(self):
        # print("==== Measure")

        qubit_outcomes = [
            self.int_to_binary(k, self.N) for k in range(2**self.N)]

        # Overall probability of a photon click at `n`.
        self.outcomes = pd.DataFrame(
            index=pd.Index(range(self.N+1), name='n'),
            columns=["probability"]+qubit_outcomes
            )
        self.outcomes["probability"] = self.outcomes.index.map(
            lambda n: self.measure_at(n=n))
        assert np.isclose(self.outcomes["probability"].sum(), 1)

        # Fine grain the probability of a photon click at `n` over the
        # different qubits.
        self.outcomes[qubit_outcomes] = pd.DataFrame(
            [qubit_outcomes] * len(self.outcomes),
            index=self.outcomes.index, columns=qubit_outcomes)
        for outcome in qubit_outcomes:
            self.outcomes[outcome] = self.outcomes.apply(
                lambda x: self.measure_at(x.name, x[outcome]), axis=1)
        assert np.isclose(
            self.outcomes[qubit_outcomes].sum(axis=1),
            self.outcomes["probability"]).all()
        self.outcomes = self.outcomes.T

    
    @staticmethod
    def rotate(psi_qubits: np.array):
        def invert(qubit: np.array):
            return np.array(
                [[np.conjugate(qubit[0]), np.conjugate(qubit[1])],
                 [-qubit[1], qubit[0]]],
                dtype=complex)        
        
        rotation = invert(psi_qubits[0])
        for q in psi_qubits[1:]:
            rotation = np.kron(rotation, invert(q))
        rotation = np.kron(rotation, np.eye(len(psi_qubits)+1))
        return rotation

    def realign(self):
        # print("==== Realign")
        rotation = self.rotate(self.psi_qubits)
        self.psi.amplitude = rotation @ self.psi.amplitude

    @staticmethod
    def symmetric_BS(N, add_qubits: bool = True):
        phi = np.exp(2*np.pi*1j/N)
        BS = np.zeros((N+1, N+1), dtype=complex)

        for n in range(N+1):
            for m in range(N+1):
                if n > 0 and m > 0:
                    BS[n, m] = phi**((n-1)*(m-1))/np.sqrt(N)
                elif n == m == 0:
                    BS[n, m] = 1
                else:
                    BS[n, m] = 0

        # Add the identity to all the qubit modes.
        if add_qubits:
            BS = np.kron(np.eye(2**N), BS)

        return BS

    @staticmethod
    def int_to_binary(k: int, N: int) -> str:
        """
        Convert integer k to a binary string of length N, using '0' and '1'
        characters.
        """
        assert 2**N >= k
        return bin(k)[2:].zfill(N)[-N:]

    @staticmethod
    def iround(c: complex, tol: float = 1e-9) -> complex:
        if abs(np.imag(c)) < tol:
            c = np.real(c)
        if abs(np.real(c)) < tol:
            c = np.imag(c)
        return c

    @staticmethod
    def norm(psi) -> float:
        return np.sqrt(psi @ np.conjugate(psi))

    def print_outcomes(self, value: tuple = None, decimals: int = 3):
    
        df = self.outcomes    

        df['T'] = df.sum(axis=1)
    
        if decimals is not None:
            df = df.round(decimals)
        
        if value is not None:
            df = df.replace(value[0], value[1])
        
        print(df)

if __name__ == "__main__":    

    for psi_qubits in ([0,1], [1,0], [1,1]):
        for N in range(2, 4):
            for realign in {True, False}:
                print(f"{psi_qubits = }, {N = }, {realign = }:")
                ifm = IFM(
                    psi_photon=[0]+[1]*N,
                    psi_qubits=[psi_qubits]*N,
                    steps=('interact', 'beam_split', 'realign' if realign else None, 'measure')
                    )
                ifm()
                psi = ifm.psi
                outcomes = ifm.outcomes
                ifm.print_outcomes((0, '◯'), decimals=3)
                print()

# %% Experiment

