# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:04:59 2023

@author: Amine Laghaout

Generalization of the Elitzur-Vaidman bomb tester to `N`-partite Mach-Zehnder
interferometers where the bombs can be in a superposition of being on a path 
and away from the path.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantum_information as qi
import seaborn as sns

class IFM:
    
    def __init__(self, bombs, gamma=1, tol = 1e-15, validate=True):
        """
        Parameters
        ----------
        bombs : tuple of np.ndarray or str
            Quantum states of the bombs. These can be specified explicitly in a
            tuple of NumPy arrays or as a string of characters where 
            - 'c' represents that the bomb is "cleared" from the photon's path, 
            - 'b' represents that the bomb is "blocking" the photon's path, and
            - 'w' represents that the bomb is in a coherent superposition of 
              being in the photon's path and away from the photon's path. 
            Each bomb lives in a 3-dimensional Hilbert space where
            - dimension 0 represents the exploded bomb,
            - dimension 1 represents the bomb in the photon's path, and
            - dimension 2 represents the bomb away from the photon's path.
            There are `N` bombs.
        gamma : np.ndarray or int, optional
            Quantum state of the incident photon. It can be specified 
            explicitly in Hilbert space as as nd.ndarray or implicitly as an 
            integer specifying which path is excited by the photon (with all
            other paths assumed to be empty.)
            The photon lives in a 3-dimensional Hilbert space where
            - dimension 0 means that the photon was absorbed by a bomb,
            - dimension `k` means that the photon travels in path `k` of the
              `N`-armed Mach-Zehnder interferometer.
        tol : float, optional
            Numerical tolerance.
        validate : bool
            Validate the math (e.g., unitarity of the beam splitter, )? 

        Returns
        -------
        None.
        """
        
        # If the bombs are represented by a string of characters and the 
        # photon by the path it is exciting, generate the corresponding state
        # vectors.
        if isinstance(bombs, str) and isinstance(gamma, int):
            bombs = tuple(self.parse_bombs(b) for b in bombs)
            gamma_temp = np.zeros(len(bombs)+1)
            gamma_temp[gamma] = 1
            gamma = gamma_temp

        # Ensure the number of bombs is equal to the number of photon paths.
        if validate: assert gamma.shape[0] == len(bombs) + 1
        for bomb in bombs: 
            if validate: assert isinstance(bomb, np.ndarray)
        
        self.validate = validate            # Validation flag
        self.tol = tol                      # Numerical tolerance
        self.gamma = gamma                  # Photonic state over the modes
        self.dims = [self.gamma.shape[0]]   # Dimensions of the Hilbert spaces
        self.bombs = bombs                  # Bomb states
        self.modes = len(self.gamma) - 1    # Number of photon paths
                
        # Construct a symmetric `N`-mode beam splitter. Note that this beam 
        # splitter has one extra dimension to account for the fact that the 
        # incident vacuum remains unchanged.
        self.BS = qi.symmetric_BS(self.modes)
        if self.validate: assert self.modes == self.BS.shape[0] - 1
        if self.validate: assert qi.is_unitary(self.BS)

        # Construct the overall bomb state vector.
        self.dims += [bomb.shape[0] for bomb in self.bombs]
        self.bombs = bombs[0]
        for k in range(self.modes - 1):
            self.bombs = np.kron(self.bombs, bombs[k+1])

        # Consrtuct the initial density matrix.
        input_state = np.kron(self.gamma, self.bombs)
        self.rho = np.outer(input_state, input_state)
        if self.validate: assert qi.is_density_matrix(self.rho)

        
    def __call__(self, interact=True, verbose=False):
        """
        Run the photon through the Mach-Zehnder interferometer.

        Parameters
        ----------
        interact : bool, optional
            Shall the photon interact with the bombs?
        verbose : bool, optional
            Print updates on the evolution of the system?

        Returns
        -------
        None.
        """

        # Report
        self.report = {
            m: {'measurement': None,    # Measurement operator
                'probability': None,    # Outcome probability
                'post_rho': None,       # Overall output state
                'subsystems': {         # Output subsystems
                    'subrho': None,     # Output subsystem states
                    'purity': None,     # Purity of subsystem state
                    'diagonals': None}  # Diagonal of subsystem states
                } for m in range(self.modes + 1)}

        #%% Initial density matrix

        if verbose:
            print('========== Initial state')
            self.print_states(self.rho, self.modes, self.dims) 
        
        #%% The first beam splitter
        
        # Generalize the beam splitter operation so that that it can act on the
        # photon-bombs system.
        BS = np.kron(self.BS, np.eye(self.bombs.shape[0]))
        if self.validate: assert qi.is_unitary(BS)
        
        # Operate the beam splitter on the state.
        self.rho = qi.BS_op(BS, self.rho)
        if self.validate: assert qi.is_density_matrix(self.rho)
        
        if verbose:
            print('\n========== After the first beam splitter')
            self.print_states(self.rho, self.modes, self.dims)    
    
        #%% The photon-bombs interactions
        
        if interact:
            # TODO: Check that the order of the interactions does not matter.
            self.interac()  
            if self.validate: assert qi.is_density_matrix(self.rho)
            
            if verbose:
                print('\n========== After the photon-bombs interactions')
                self.print_states(self.rho, self.modes, self.dims)
            
        #%% The second beam splitter
        
        self.rho = qi.BS_op(BS, self.rho)
        if self.validate: assert qi.is_density_matrix(self.rho)
        
        if verbose:
            print('\n========== After the second beam splitter')
            self.print_states(self.rho, self.modes, self.dims)    
    
        #%% Photon measurements in the Fock basis
        
        print('\n========== After the photon measurements')
                
        # For each measurement output…
        for k in self.report.keys():
            
            #### Measurement operator
            
            # The projection operators as state vectors are all initialized to 
            # 0. There are as many projection operators as the dimension of the 
            # Hilbert space, which---recall---also include a projection on the 
            # vacuum, corresponding to the absorption of the photon upon the 
            # explosion of the bomb.
            measurement = np.zeros(self.modes + 1)
            # Select the dimension of the Hilbert space to be projected on.
            measurement[k] = 1
            # Construct the corresponding measurement operator.
            measurement = np.outer(measurement, measurement)
            # The Hilbert space of the bombs is not projected upon, so just 
            # expand the measurement operator with the identity.
            measurement = np.kron(measurement, np.eye(len(self.bombs)))
            # Check the Hermicity of the measurement operator thus constructed.
            if self.validate: assert qi.is_Hermitian(measurement)
            self.report[k]['measurement'] = measurement
            
            #### Probability of the measurement
            
            probability = qi.Born(measurement, self.rho)
            self.report[k]['probability'] = probability
            
            #### Post-measurement state
            
            if probability > self.tol:
                post_rho = measurement @ self.rho @ measurement/probability
                if self.validate: assert qi.is_density_matrix(post_rho)    
                self.report[k]['post_rho'] = qi.trim_imaginary(post_rho)

            print(f"\n===== Prob({k}): {np.round(probability,3)} =====")
            self.report[k]['subsystems'] = self.print_states(post_rho, self.modes, self.dims, True)
            
        probabilities = [self.report[k]['probability'] for k in range(self.modes+1)]
        assert -self.tol <= sum(probabilities) <= 1 + self.tol
        self.plot_probabilities(probabilities)

    @staticmethod
    def parse_bombs(bomb):
        # 'C'leared: The is a bomb remotely and the vacuum locally.
        if bomb.upper() == 'C':
            return np.array([0, 0, 1])
        # 'B'locked: There is a bomb locally and the vacuum remotely.
        elif bomb.upper() == 'B': 
            return np.array([0, 1, 0])
        # 'E'qual: There is a bomb that in a superposition of being local and 
        # remote.
        elif bomb.upper() == 'E':
            return np.array([0, 1, 1])/np.sqrt(2)
        else:
            raise ValueError("Wrong letter coding for the bomb")
    
    def interac(self):  # TODO: Check this function
        
        rho = self.rho.copy()  

        for mode in range(1, self.modes + 1):
                   
            # Transitions
            starting_states = self.interac_helper(mode, False)        
            ending_states = self.interac_helper(mode, True)            
            state_transitions = list(zip(starting_states, ending_states))
            # print(f'Interaction at mode {mode}:')
            # print(state_transitions)
            
            C = np.eye(self.rho.shape[0])
            
            for trans in state_transitions:
                S = np.zeros(self.rho.shape)
                S[trans[0], trans[0]] = 1
                P = np.eye(self.rho.shape[0])
                row = P[trans[0], :].copy()
                P[trans[0], :] = P[trans[1], :]
                P[trans[1], :] = row
               
                C[trans[0], :] = 0
                
                rho += P @ S @ rho + rho @ S.conj().T @ P.conj().T + \
                    P @ S @ rho @ S.conj().T @ P.conj().T
                
                rho = C @ rho @ C.conj().T
                
            self.rho = rho
        
    def interac_helper(self, mode, status):
        # TODO: What is going on here?
        if status is False:
            state = np.array([0, 1, 0])
        else:
            state = np.array([1, 0, 0])
        
        bomb = np.array([1, 1, 1])
        bombs = bomb
        if mode==1:
            bombs = state
        for m in range(1, self.modes):
            if m == mode - 1:
                bombs = np.kron(bombs, state)
            else:
                bombs = np.kron(bombs, bomb)
            
        gamma = np.zeros(self.modes + 1)
        if status is False:
            gamma[mode] = 1
        else:
            gamma[0] = 1
        
        return np.where(np.kron(gamma, bombs) == 1)[0]

    def print_states(self, rho, modes, dims, save=True):
        
        subsystems = {}
    
        if rho is None:
            print('Impossible scenario.')
        else:
            subrho = qi.trim_imaginary(qi.partial_trace(rho, [0], dims))
            if self.validate: assert qi.is_density_matrix(subrho)
            purity = qi.purity(subrho, self.tol)
            print('photon, purity:', np.round(purity, 4)) 
            print(np.round(subrho, 3)) 
            subsystems['photon'] = {i: d for i, d in enumerate(subrho.diagonal())}
            subsystems['photon'].update({'purity': purity, 'subrho': subrho})
            for k in range(modes):
                subrho = qi.trim_imaginary(qi.partial_trace(rho, [k+1], dims))
                if self.validate: assert qi.is_density_matrix(subrho)
                purity = qi.purity(subrho, self.tol)
                print(f'bomb {k+1}, purity:', np.round(purity, 4))
                print(np.round(subrho, 3))
                subsystems[f'bomb {k+1}:'] = {i: d for i, d in enumerate(subrho.diagonal())}
                subsystems[f'bomb {k+1}:'].update({'purity': purity, 'subrho': subrho})
                
        return subsystems
        
    @staticmethod
    def plot_probabilities(values):
        
        # Extract keys and values from the dictionary
        # keys = list(data_dict.keys())
        # values = list(data_dict.values())
        
        # Create a Seaborn histogram
        sns.set(style="whitegrid", 
                rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15})
        # plt.figure(figsize=(10, 6))  # Set the figure size (adjust as needed)
        
        # Create the histogram using Seaborn
        sns.barplot(x=list(range(len(values))), y=values, 
                    palette=['darkred']+['darkblue']*(len(values)-1))
        
        # Set labels and title
        plt.xlabel('Mode')
        plt.ylabel('Probability')
        plt.title('Path-encoded measurement probabilities')
        
        # Show the plot
        # plt.xticks()  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Ensure all elements are visible
        plt.show()


#%% Run as a script, not as a module.
       
if __name__ == "__main__":
    my_system = IFM('ecce', 1)
    my_system(verbose=True)
    report = my_system.report