# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:52:40 2025

@author: amine
"""

from pydantic import BaseModel
from typing import Optional
import numpy as np


class IFM(BaseModel):
    N: Optional[int] = None
    psi_photon: Optional[object] = None
    psi_qubits: Optional[object] = None

    def __call__(self):
        
        self.validate()
        print(self.N)
    
    def validate(self):
        if self.N is None:
            self.N = len(self.psi_photon)
        assert self.N == len(self.psi_photon)
        assert self.psi_qubits.shape[0] == self.N
        assert self.psi_qubits.shape[1] == 2
        
        # Normalize the photonic state.
        norm_photon = np.sqrt(self.psi_photon @ np.conjugate(self.psi_photon))
        self.psi_photon = self.psi_photon / norm_photon
        norm_photon = np.sqrt(self.psi_photon @ np.conjugate(self.psi_photon))
        assert np.isclose(norm_photon, 1)
        
        # Normalize the qubits.
        # norm_qubits = np.sqrt(self.psi_qubits * np.conjugate(self.psi_qubits))
        # self.psi_qubits = self.psi_qubits / norm_qubits        
        # norm_qubits = np.sqrt(self.psi_qubits * np.conjugate(self.psi_qubits))        
        # assert np.isclose(norm_qubits, 1)
    
A = IFM(psi_photon=np.ones(3),
        psi_qubits=np.random.rand(3, 2))
A()

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
        
    