import numpy as np
from scipy.optimize import minimize

# Example data
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Coefficient matrix
b = np.array([1, 2, 3])  # Right-hand side


# Objective function: Sum of squared residuals
def objective(x):
    return np.sum((A.dot(x) - b) ** 2)


# Initial guess for the coefficients
x0 = np.random.rand(A.shape[1])

# Constraints: coefficients between 0 and 1, and sum to unity
cons = (
    {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Sum to unity
    {"type": "ineq", "fun": lambda x: x},  # Greater than 0
    {"type": "ineq", "fun": lambda x: 1 - x},
)  # Less than 1

# Solve the constrained optimization problem
result = minimize(objective, x0, constraints=cons)

# The optimized coefficients
print("Optimized coefficients:", result.x)
