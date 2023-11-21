# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:04:59 2023

@author: amine
"""

import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-14

class IFM:
    
    def __init__(
            self, 
            # Two modes
            # gamma=np.array([0,1,0]), 
            # bombs=(np.array([0,0,1]),)*2,  # all cleared
            # bombs=(np.array([0,1,0]),)*2,  # all blocked
            # bombs=(np.array([0,0,1]), np.array([0,1,0])),  # cleared-blocked
            # bombs=(np.array([0,1,0]), np.array([0,0,1])),  # blocked-cleared
            # bombs=(np.array([0,1,1])/np.sqrt(2), np.array([0,0,1])),  # equal-cleared
            # bombs=(np.array([0,0,1]), np.array([0,1,1])/np.sqrt(2)),  # cleared-equal
            # bombs=(np.array([0,1,1])/np.sqrt(2),)*2,  # Equal-equal
            # Three modes
            gamma=np.array([0,1,0,0]), 
            # bombs=(np.array([0,0,1]),)*3,  # All-cleared
            # bombs=(np.array([0,1,0]),)*3,  # All-blocked
            # bombs=(np.array([0,1,1])/np.sqrt(2),)*2+(np.array([0,0,1]),),  # equal-equal-cleared
            # bombs=(np.array([0,1,1])/np.sqrt(2),)+(np.array([0,0,1]),)*2,  # equal-cleared-cleared
            # bombs=(np.array([0,1,0]),)+(np.array([0,0,1]),)*2,  # blocked-cleared-cleared
            # bombs=(np.array([0,1,0]),)*2+(np.array([0,0,1]),),  # blocked-blocked-cleard
            bombs=(np.array([0,1,1])/np.sqrt(2),)*3,  # equal-equal-equal
            ):
        
        self.dims = []  # Dimension of the Hilbert subspaces
        self.gamma = gamma  # Photonic state over the modes
        self.dims += [self.gamma.shape[0]] 
        self.bombs = bombs  # Bomb states
        self.modes = len(self.gamma) - 1  # Number of interferometer arms

        if self.modes == 2:
            self.BS = np.array([[np.sqrt(2), 0, 0], 
                                [0, 1, 1], 
                                [0, 1, -1]])/np.sqrt(2)
        elif self.modes == 3:
            alpha = (-1 + 1j*np.sqrt(3))/2
            self.BS = np.array([[np.sqrt(3), 0, 0, 0], 
                                [0, 1, 1, 1], 
                                [0, 1, alpha, alpha**2], 
                                [0, 1, alpha**2, alpha**4]])/np.sqrt(3)            

        assert self.modes == self.BS.shape[0] - 1
        assert self.is_unitary(self.BS)

        # Construct the bomb state vector.
        self.dims += [bomb.shape[0] for bomb in self.bombs]
        self.bombs = bombs[0]
        for k in range(self.modes - 1):
            self.bombs = np.kron(self.bombs, bombs[k+1])
        
    def __call__(self):

        #%% Initial state
        print('========== Initial state')
        self.input_state = np.kron(self.gamma, self.bombs)
        self.rho = np.outer(self.input_state, self.input_state)
        # print('>>', self.rho.shape, self.dims)
        # print(self.gamma.shape, self.bombs.shape, self.input_state.shape)
        assert self.is_unitary(self.BS) and self.is_density_matrix(self.rho)
        
        self.plot(self.rho, 'Initial state')   
        self.print_states(self.rho, self.modes, self.dims)    
        
        #%% After the first beam splitter
        print('\n========== After the first beam splitter')
        BS = np.kron(self.BS, np.eye(self.bombs.shape[0]))
        assert self.is_unitary(BS)
        self.rho = BS @ self.rho @ np.conjugate(BS.T)
        assert self.is_density_matrix(self.rho)
        self.plot(self.rho, 'After the first beam splitter')
        self.print_states(self.rho, self.modes, self.dims)    
    
        #%% After the interactions
        print('\n========== After the interactions')
        self.interac()
        assert self.is_density_matrix(self.rho)
        self.print_states(self.rho, self.modes, self.dims)    
            
        #%% After the second beam splitter
        print('\n========== After the second beam splitter')
        self.rho = BS @ self.rho @ np.conjugate(BS.T)
        assert self.is_density_matrix(self.rho)
        self.plot(self.rho, 'After the second beam splitter') 
        self.print_states(self.rho, self.modes, self.dims)    
    
        #%% After the measurements
        print('\n========== After the measurements')
        self.measurements = {
            i: np.zeros(self.modes + 1) for i in range(self.modes + 1)}
        
        # Form all the possible measurement operators.
        for k in self.measurements.keys():
            self.measurements[k][k] = 1
            self.measurements[k] = np.outer(self.measurements[k], 
                                            self.measurements[k])
            self.measurements[k] = np.kron(self.measurements[k], np.eye(len(self.bombs)))

        # TODO: Check the validity of the measurement operator.

        # Compute the probabilities for each measurement.
        self.probabilities = {m: np.trace(self.measurements[m] @ self.rho) 
                              for m in self.measurements.keys()}
        assert abs(1 - sum(self.probabilities.values())) < TOL        
        
        # Compute the post-measurement states for each measurement.
        self.post_rho = {m: None for m in self.measurements.keys()}
        for m in self.post_rho.keys():           
            if self.probabilities[m] > TOL:
                self.post_rho[m] = self.measurements[m] @ self.rho @ self.measurements[m]/self.probabilities[m]
                assert self.is_density_matrix(self.post_rho[m])                
            else:
                self.post_rho[m] = None

            print(f"\n===== Prob({m}): {np.round(self.probabilities[m],3)} =====")
            self.print_states(self.post_rho[m], self.modes, self.dims)
        
        
    
    def interac(self):

        rho = self.rho.copy()  # TODO: Move outside the loop.        

        for mode in range(1, self.modes + 1):

            print(f'Interaction at mode {mode}:')
                   
            # Transitions
            starting_states = self.interac_helper(mode, False)        
            ending_states = self.interac_helper(mode, True)            
            state_transitions = list(zip(starting_states, ending_states))
            print(state_transitions)
            
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
        
    def is_density_matrix(self, rho, tol=TOL):
        
        """ Check that the matrix is Hermitian and of trace one. """
                
        return abs(1 - np.trace(rho)) < tol and np.allclose(rho, np.conjugate(rho.T), atol=tol) and np.all(np.linalg.eigvalsh(rho) >= -tol)


    def is_unitary(self, matrix, tol=TOL):
        # Calculate the conjugate transpose (Hermitian)
        conj_transpose = np.conjugate(matrix.T)
        
        # Perform the multiplication of the matrix with its conjugate transpose
        product = np.dot(matrix, conj_transpose)
        
        # Check if the product is close to the identity matrix
        identity = np.eye(matrix.shape[0])
        
        # Use np.allclose to check if matrices are close within the specified tolerance
        return np.allclose(product, identity, atol=tol)

    def print_states(self, rho, modes, dims):
    
        if rho is None:
            print('Impossible scenario.')
        else:
            print('Photon:')
            print(np.round(self.partial_trace(rho, [0], dims), 3))
            for k in range(modes):
                print(f'Bomb {k+1}:')
                print(np.round(self.partial_trace(rho, [k+1], dims), 3))
    
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
    
        # X = [np.random.randn(3,3), # 0
        #      np.random.randn(4,4), # 1
        #      np.random.randn(2,2), # 2
        #      np.random.randn(3,3), # 3
        #      np.random.randn(7,7), # 4
        #      ]
        # X = [(10*x).round() for x in X]
        # Y = X[0]
        # for k in range(len(X)-1):
        #     Y = np.kron(Y, X[k+1])
        # print(partial_trace(Y, [1], [3,4,2,3,7]))

        
my_system = IFM()
my_system()
measurements = my_system.measurements
probabilities = my_system.probabilities
post_rho = my_system.post_rho