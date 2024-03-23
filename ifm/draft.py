import numpy as np
from scipy.optimize import minimize

# Example linear system (A and b)
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2, 3])


# Objective function: Least squares
def objective(x):
    return np.sum((A.dot(x) - b) ** 2)


# Constraints
cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Sum to unity
bounds = [(0, 1), (0, 1)]  # Probability constraints for each element of x

# Initial guess
x0 = np.random.rand(A.shape[1])
x0 /= np.sum(x0)  # Normalize to satisfy the sum to unity constraint initially

# Solve the constrained optimization problem
result = minimize(objective, x0, bounds=bounds, constraints=cons)

if result.success:
    print("Optimal solution found:", result.x)
else:
    print("Optimization failed.")
