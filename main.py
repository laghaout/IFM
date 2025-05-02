# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:52:40 2025

@author: amine
"""

from functools import reduce
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional


class IFM(BaseModel):
    N: Optional[int] = None                 # Number of qubits
    psi_photon: Optional[object] = None     # Photonic state
    psi_qubits: Optional[object] = None     # Qubit states
    psi: Optional[object] = None            # Overall state

    def __call__(self):

        # If the number of qubits is specified, use an equal distribution of
        # the photonic and qubit modes. This shall be the default.
        if isinstance(self.N, int):
            self.psi_photon = np.hstack([np.zeros(1), np.ones(self.N)])
            self.psi_qubits = np.ones([self.N, 2])
        else:
            assert len(self.psi_photon) == self.psi_qubits.shape[0]
            self.N = len(self.psi_photon)

        self.validate()

        print("psi_photon =\n", self.psi_photon)
        print("psi_qubits =\n", self.psi_qubits)

        self.psi_qubits = reduce(np.kron, self.psi_qubits)
        self.psi = np.kron(self.psi_qubits, self.psi_photon)

        max_k = 2**(self.N)
        index_tuples = [(k, n) for k in range(max_k)
                        for n in range(self.N + 1)]
        index = pd.MultiIndex.from_tuples(index_tuples, names=["k", "n"])
        self.psi = pd.DataFrame(
            index=index, data={"amplitude": self.psi})
        
        self.psi['ket'] = self.psi.index.map(
            lambda k: self.int_to_binary(k[0], self.N)+'⊗'+str(k[1]))

        self.interact()

        self.beam_split()
        
        self.measure()

    def validate(self, normalize=True):

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

        if normalize:
            # Normalize the qubits.
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
        self.psi.amplitude = self.psi.amplitude

    def beam_split(self):
        self.psi.amplitude = self.symmetric_BS(self.N) @ self.psi.amplitude
    
    def measure(self):
        for n in range(self.N+1):
            mask = self.psi.index.map(lambda x: 1 if x[1] == n else 0)
            amplitude = mask * self.psi.amplitude
            probability = amplitude @ np.conjugate(amplitude)
            print(n, probability)

    @staticmethod
    def symmetric_BS(N):
        phi = np.exp(2*np.pi*1j/N)
        BS = np.empty((N+1, N+1), dtype=complex)

        for n in range(N+1):
            for m in range(N+1):
                if n > 0 and m > 0:
                    BS[n, m] = phi**((n-1)*(m-1))
                elif n == m == 0:
                    BS[n, m] = 1
                else:
                    BS[n, m] = 0

        BS = np.kron(np.eye(2**N), BS)

        return BS

    @staticmethod
    def int_to_binary(k, N):
        """
        Convert integer k to a binary string of length N,
        using '0' and '1' characters.
        """
        return bin(k)[2:].zfill(N)[-N:]

A = IFM(N=2)
A()
psi = A.psi
BS = A.symmetric_BS(A.N)


#%%

