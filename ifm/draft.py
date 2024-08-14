# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:38:02 2024

@author: amine
"""

import sympy as sp

# Define the vectors a and b
a = sp.Matrix([1, 2])
b = sp.Matrix([3, 4])

# Compute the tensor product (Kronecker product)
tensor_product = sp.tensorproduct(a, b)

# Convert the result to a vector by flattening the matrix
# Instead of using sp.Matrix(tensor_product).vec(), we'll directly reshape the matrix.
tensor_product_vector = tensor_product.reshape(len(a) * len(b), 1)

# Display the vector
tensor_product_vector
