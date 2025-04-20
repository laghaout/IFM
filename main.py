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
        # the photonic and qubit modes. This shall serve as the default.
        if isinstance(self.N, int):
            self.psi_photon = np.hstack([np.zeros(1), np.ones(self.N)])
            self.psi_qubits = np.ones([self.N, 2])
        else:
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

    @staticmethod
    def symmetric_BS(N):
        phi = np.exp(1j*np.pi/N)
        return phi


A = IFM(N=3)
A()

# for k in 

# def foo(i, n, N, phi, P, Z):
#     """
#     Probability of ⟨bin_N(i)|
#     """
#     def inner_sum(j, phi, N):
#         return sum(
#             phi**((n-1)*(m-1))*kappa_prime(m, j, N)
#             for m in range(1, N+1))
    
#     def outer_sum(N, P, Z):
#         total = sum(
#             Z[i, j]*inner_sum(j, phi, N)/(P[n]*np.sqrt(N))
#             for j in range(1, 2**(N-1)-1))
#         return total*np.conjugate(total)
        
    