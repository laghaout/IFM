# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:04:30 2024

@author: amine
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import qutip.visualization as qtv


N = 20

rho_coherent = qt.coherent_dm(N, np.sqrt(2))
# rho_thermal = qt.thermal_dm(N, 2)
# rho_fock = qt.fock_dm(N, 2)
# rho_fock = qt.Qobj(np.array([[1, -1j], [1j, 1]])/2)


rho_thermal = report.actual.rho.final.loc[(1, 1)]
rho_fock = report.actual.rho.final.loc[(2, 1)]
rho_fock = report.actual.rho.final.loc[(3, 1)]


fig, axes = plt.subplots(1, 3, figsize=(12, 3))
qtv.plot_fock_distribution(
    rho_coherent, fig=fig, ax=axes[0], title="Coherent state"
)
qtv.plot_fock_distribution(
    rho_thermal, fig=fig, ax=axes[1], title="Thermal state"
)
qtv.plot_fock_distribution(rho_fock, fig=fig, ax=axes[2], title="Fock state")
fig.tight_layout()
plt.show()


xvec = np.linspace(-5, 5, 500)
W_coherent = qt.wigner(rho_coherent, xvec, xvec)
W_thermal = qt.wigner(rho_thermal, xvec, xvec)
W_fock = qt.wigner(rho_fock, xvec, xvec)

# plot the results
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
cont0 = axes[0].contourf(xvec, xvec, W_coherent, 100)
lbl0 = axes[0].set_title("Coherent state")
cont1 = axes[1].contourf(xvec, xvec, W_thermal, 100)
lbl1 = axes[1].set_title("Thermal state")
cont0 = axes[2].contourf(xvec, xvec, W_fock, 100)
lbl2 = axes[2].set_title("Fock state")
plt.show()
