# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:38:02 2024

@author: amine
"""

import numpy as np
from sympy import symbols, Matrix

# Create a SymPy symbol for the Greek gamma (γ)
gamma = symbols("γ")

# Define the dimensions of the matrix
rows = 4  # Number of rows (n)
cols = 4  # Number of columns (m)

# Create the SymPy matrix using list comprehensions
sympy_matrix = Matrix(
    [
        [gamma ** (i * j) for j in range(1, cols + 1)]
        for i in range(1, rows + 1)
    ]
)

# Print the SymPy matrix
print("SymPy Matrix:")
print(sympy_matrix)

# Convert the SymPy matrix to a NumPy array of objects
numpy_array = np.array(sympy_matrix.tolist(), dtype=object)

# Print the NumPy array
print("\nNumPy Array (with symbolic expressions):")
print(numpy_array)
