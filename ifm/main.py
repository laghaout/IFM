# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:04:59 2023

@author: Amine Laghaout
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantum_information as qi

class IFM:
    
    def __init__(self, bombs, gamma=1, tol = 1e-14):
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
        assert gamma.shape[0] == len(bombs) + 1
        for bomb in bombs: 
            assert isinstance(bomb, np.ndarray)
        
        self.tol = tol                      # Numerical tolerance
        self.gamma = gamma                  # Photonic state over the modes
        self.dims = [self.gamma.shape[0]]   # Dimensions of the Hilbert spaces
        self.bombs = bombs                  # Bomb states
        self.modes = len(self.gamma) - 1    # Number of photon paths
        
        # Construct a symmetric `N`-mode beam splitter. Note that this beam 
        # splitter has one extra dimension to account for the fact that the 
        # incident vacuum remains unchanged.
        self.BS = self.symmetric_BS(self.modes)
        assert self.modes == self.BS.shape[0] - 1
        assert self.is_unitary(self.BS)

        # Construct the overall bomb state vector.
        self.dims += [bomb.shape[0] for bomb in self.bombs]
        self.bombs = bombs[0]
        for k in range(self.modes - 1):
            self.bombs = np.kron(self.bombs, bombs[k+1])

        # Consrtuct the initial density matrix.
        input_state = np.kron(self.gamma, self.bombs)
        self.rho = np.outer(input_state, input_state)
        assert self.is_density_matrix(self.rho)
        
    def __call__(self, interact=True, verbose=False):

        #%% Initial density matrix

        if verbose:
            print('========== Initial state')
            self.print_states(self.rho, self.modes, self.dims)   
        
        #%% After the first beam splitter
        
        BS = np.kron(self.BS, np.eye(self.bombs.shape[0]))
        assert self.is_unitary(BS)
        self.rho = BS @ self.rho @ np.conjugate(BS.T)
        assert self.is_density_matrix(self.rho)
        
        if verbose:
            print('\n========== After the first beam splitter')
            self.print_states(self.rho, self.modes, self.dims)    
    
        #%% After the interactions
        
        if interact:
            # TODO: Check the order of the interactions does not matter.
            self.interac()  
            assert self.is_density_matrix(self.rho)
            
            if verbose:
                print('\n========== After the interactions')
                self.print_states(self.rho, self.modes, self.dims)
            
        #%% After the second beam splitter
        
        self.rho = BS @ self.rho @ np.conjugate(BS.T)
        assert self.is_density_matrix(self.rho)
        
        if verbose:
            print('\n========== After the second beam splitter')
            self.print_states(self.rho, self.modes, self.dims)    
    
        #%% After the measurements
        
        print('\n========== After the measurements')
        
        # Compute photon measurement operators in the Fock basis.
        self.measurements = {
            i: np.zeros(self.modes + 1) for i in range(self.modes + 1)}
        for k in self.measurements.keys():
            # Pick the `k`-th mode of the photon.
            self.measurements[k][k] = 1
            self.measurements[k] = np.outer(self.measurements[k], 
                                            self.measurements[k])
            # The Hilbert space of the bombs is not projected upon.
            self.measurements[k] = np.kron(
                self.measurements[k], np.eye(len(self.bombs)))

            # TODO: Check the validity of the measurement operators.

        # Compute the probabilities for each measurement.
        self.probabilities = {m: np.trace(self.measurements[m] @ self.rho) 
                              for m in self.measurements.keys()}
        assert abs(1 - sum(self.probabilities.values())) < self.tol   
        self.plot_probabilities(self.probabilities)
        
        # Compute the post-measurement states for each measurement.
        self.post_rho = {m: None for m in self.measurements.keys()}
        for m in self.post_rho.keys():           
            if self.probabilities[m] > self.tol:
                self.post_rho[m] = self.measurements[m] @ self.rho @ self.measurements[m]/self.probabilities[m]
                assert self.is_density_matrix(self.post_rho[m])                
            else:
                self.post_rho[m] = None

            print(f"\n===== Prob({m}): {np.round(self.probabilities[m],3)} =====")
            self.print_states(self.post_rho[m], self.modes, self.dims, True)

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
            assert False, "Wrong letter coding for the bomb"
        
    @staticmethod        
    def symmetric_BS(N, add_explosion_mode=True):
        
        a = 2*np.pi/N
        a = np.cos(a) + 1j*np.sin(a)    
        BS = np.array([[a**(r*c) for c in range(N)] for r in range(N)])
        if add_explosion_mode:
            BS = np.concatenate((np.zeros((N, 1)), BS), axis=1)
            BS = np.concatenate((np.zeros((1, N+1)), BS), axis=0)
            BS[0,0] = np.sqrt(N)
        BS /=  np.sqrt(N)
        return BS       
    
    def interac(self):

        rho = self.rho.copy()  # TODO: Move outside the loop.        

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
            
            assert self.is_density_matrix(rho)
            
        
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

    def plot(self, rho, title=None):
        
        cells = rho.shape[0]
        
        # Use Matplotlib to render the array with a grid
        fig, ax = plt.subplots()
        # TODO: Should I also draw the imaginary part?
        cax = ax.matshow(rho.real, cmap='viridis')  
        
        # Add color bar
        plt.colorbar(cax)
        
        # Set ticks
        ax.set_xticks(np.arange(-0.5, cells, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, cells, 1), minor=True)
        
        # Gridlines based on minor ticks
        ax.grid(which="minor", color="w", linestyle='-', linewidth=.08)
        
        # Hide major tick labels
        ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
        
        if isinstance(title, str):
            plt.title(title)
        plt.show()
        
    def is_density_matrix(self, rho):
        
        """ Check that the matrix is Hermitian and of trace one. """
                
        return abs(1 - np.trace(rho)) < self.tol and np.allclose(rho, np.conjugate(rho.T), atol=self.tol) and np.all(np.linalg.eigvalsh(rho) >= -self.tol)


    def is_unitary(self, matrix):
        # Calculate the conjugate transpose (Hermitian)
        conj_transpose = np.conjugate(matrix.T)
        
        # Perform the multiplication of the matrix with its conjugate transpose
        product = np.dot(matrix, conj_transpose)
        
        # Check if the product is close to the identity matrix
        identity = np.eye(matrix.shape[0])
        
        # Use np.allclose to check if matrices are close within the specified tolerance
        return np.allclose(product, identity, atol=self.tol)

    def print_states(self, rho, modes, dims, save=True):
    
        # TODO: Save the states by returning them.
    
        self.subrho = {}
    
        if rho is None:
            print('Impossible scenario.')
        else:
            subrho = self.partial_trace(rho, [0], dims)
            assert self.is_density_matrix(subrho)
            purity = self.purity(subrho, self.tol)
            print('photon:', np.round(purity, 4)) 
            print(np.round(subrho, 3))
            self.subrho['photon'] = {i: d for i, d in enumerate(subrho.diagonal())}
            self.subrho['photon'].update({'purity': purity})
            for k in range(modes):
                subrho = self.partial_trace(rho, [k+1], dims)
                assert self.is_density_matrix(subrho)
                purity = self.purity(subrho, self.tol)
                print(f'bomb {k+1}:', np.round(purity, 4))
                print(np.round(subrho, 3))
                self.subrho[f'bomb {k+1}:'] = {i: d for i, d in enumerate(subrho.diagonal())}
                self.subrho[f'bomb {k+1}:'].update({'purity': purity})
                
            self.subrho = pd.DataFrame(self.subrho).transpose()


    
    @staticmethod
    def purity(rho, tol):
        
        purity = np.trace(rho @ rho)
        if purity.imag < tol:
            purity = purity.real
        return purity
    
    @staticmethod
    def partial_trace(rho, keep, dims, optimize=False):
        """Calculate the partial trace
    
        ρ_a = Tr_b(ρ)
    
        Parameters
        ----------
        ρ : 2D array
            Matrix to trace
        keep : array
            An array of indices of the spaces to keep after
            being traced. For instance, if the space is
            A x B x C x D and we want to trace out B and D,
            keep = [0,2]
        dims : array
            An array of the dimensions of each space.
            For instance, if the space is A x B x C x D,
            dims = [dim_A, dim_B, dim_C, dim_D]
    
        Returns
        -------
        ρ_a : 2D array
            Traced matrix
        """
        keep = np.asarray(keep)
        dims = np.asarray(dims)
        Ndim = dims.size
        Nkeep = np.prod(dims[keep])
    
        idx1 = [i for i in range(Ndim)]
        idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
        rho_a = rho.reshape(np.tile(dims,2))
        rho_a = np.einsum(rho_a, idx1+idx2, optimize=optimize)
        
        return rho_a.reshape(Nkeep, Nkeep)
        
    @staticmethod
    def plot_probabilities(data_dict):
    
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Extract keys and values from the dictionary
        keys = list(data_dict.keys())
        values = list(data_dict.values())
        
        # Create a Seaborn histogram
        sns.set(style="whitegrid", 
                rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15})
        # plt.figure(figsize=(10, 6))  # Set the figure size (adjust as needed)
        
        # Create the histogram using Seaborn
        sns.barplot(x=keys, y=values, 
                    palette=['darkred']+['darkblue']*(len(data_dict)-1))
        
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
    my_system = IFM('bc', 1)
    my_system()
    measurements = my_system.measurements
    probabilities = my_system.probabilities
    post_rho = my_system.post_rho
    subrho = my_system.subrho