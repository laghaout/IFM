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
    N: Optional[int] = None                         # Number of qubits
    psi_photon: Optional[object] = None             # Initial photonic state
    psi_qubits: Optional[object] = None             # Initial qubit states
    psi: Optional[object] = None                    # Overall state
    sep: str = '⊗'                                  # Qubit-photon separator
    interaction: str | None = 'Elitzur-Vaidman'     # Photon-qubit interaction
    outcomes: object = None                         # Summary of outcomes

    def __call__(self):

        # By default, if the number of qubits is specified, use an equal
        # distribution of the photonic and qubit modes.
        if isinstance(self.N, int):
            self.psi_photon = np.hstack([np.zeros(1), np.ones(self.N)])
            self.psi_qubits = np.ones([self.N, 2])
        else:
            assert len(self.psi_photon) == self.psi_qubits.shape[0] + 1
            self.N = len(self.psi_qubits)

        self.validate()

        print("psi_photon =", self.psi_photon, sep='\n')
        print("psi_qubits =", self.psi_qubits, sep='\n')

        self.psi_qubits = reduce(np.kron, self.psi_qubits)
        self.psi = np.kron(self.psi_qubits, self.psi_photon)

        # Assemble the overall state.
        max_k = 2**(self.N)
        index_tuples = [(k, n) for k in range(max_k)
                        for n in range(self.N + 1)]
        index = pd.MultiIndex.from_tuples(index_tuples, names=["k", "n"])
        self.psi = pd.DataFrame(index=index, data={"amplitude": self.psi})
        self.psi['ket'] = self.psi.index.map(
            lambda k: self.int_to_binary(k[0], self.N)+self.sep+str(k[1]))

        self.interact()         # The photon and qubit states interact.
        self.beam_split()       # The photon undergoes a beam splitting.
        self.measure()          # Measure.
        self.realign()          # The qubits are realigned.
        self.measure()          # Measure.

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
            print("Normalizing the photonic state.")

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
            print("Normalizing the qubits states.")

    def interact(self):

        print(f"==== Interact ({self.interaction})")

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
        print("Norm before normalization:", norm)
        self.psi.amplitude = self.psi.amplitude / norm
        norm = self.norm(self.psi.amplitude)
        print("Norm after normalization:", norm)

    def beam_split(self):
        print("==== Beam-split")
        self.psi.amplitude = self.symmetric_BS(self.N) @ self.psi.amplitude

    def measure_at(self, n: int = None, qubits: str = None) -> float:
        mask = np.ones(len(self.psi))
        if n is not None:
            mask *= self.psi.index.map(lambda x: 1 if x[1] == n else 0)
        if qubits is not None:
            mask *= self.psi.ket.apply(
                lambda x: 1 if x.split(self.sep)[0] == qubits else 0)
        amplitude = mask * self.psi.amplitude
        probability = amplitude @ np.conjugate(amplitude)
        return self.iround(probability)

    def measure(self):
        print("==== Measure")

        qubit_outcomes = [
            self.int_to_binary(k, self.N) for k in range(2**self.N)]

        self.outcomes = pd.DataFrame(
            index=pd.Index(range(self.N+1), name='n'),
            columns=["probability"]+qubit_outcomes
            )
        self.outcomes["probability"] = self.outcomes.index.map(
            lambda n: self.measure_at(n=n))
        
        assert np.isclose(self.outcomes["probability"].sum(), 1)
        
        self.outcomes[qubit_outcomes] = pd.DataFrame(
            [qubit_outcomes] * len(self.outcomes), 
            index=self.outcomes.index, columns=qubit_outcomes)
        
        for outcome in qubit_outcomes:
            self.outcomes[outcome] = self.outcomes.apply(
                lambda x: self.measure_at(x.name, x[outcome]), axis=1)
        
        assert np.isclose(
            self.outcomes[qubit_outcomes].sum(axis=1),
            self.outcomes["probability"]).all()

    def realign(self):
        print("==== Measure")
        

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
    def iround(c: complex, tol=1e-9) -> complex:
        if abs(np.imag(c)) < tol:
            c = np.real(c)
        if abs(np.real(c)) < tol:
            c = np.imag(c)
        return c
    
    @staticmethod
    def norm(psi) -> float:
        return np.sqrt(psi @ np.conjugate(psi))

#%% Experiment

if __name__ == "__main__":
    params = dict(
        psi_photon=np.array([int(k) for k in '0111']),
        psi_qubits=np.array(
            [int(k) for k in ''.join(['01', '10', '01'])]).reshape(3,2))
    # params = dict(N=3)

    ifm = IFM(**params)
    ifm()
    psi = ifm.psi
    outcomes = ifm.outcomes
    
    print(outcomes)