import numpy as np
import pandas as pd

N = 3
rangeN = list(range(1, N + 1))
index = pd.MultiIndex.from_product(
    [rangeN, rangeN], names=["level1", "level2"]
)
columns = [
    (
        "col_B",
        "col_B.1",
    ),
    (
        "col_B",
        "col_B.2",
    ),
]
components = range(1, 3)
columns += [("col_A", "col_A.1", f"col_A.1.{c}") for c in components]
columns += [("col_A", "col_A.2", f"col_A.2.{c}") for c in components]
columns = pd.MultiIndex.from_tuples(columns)
df = pd.DataFrame(
    np.random.randint(0, 9, size=(9, 6)), columns=columns, index=index
)

# df.loc[:, ("col_B", "col_B.2",)] = 7  # Warning
# df.loc[:, ("col_B", "col_B.2", slice(None))] = 7  # No warning

print("The whole df:\n", df)  # No warning
print("A subset of the df:\n", df.loc[:, ("col_A")])  # No warning
print("A subsubset of the df:\n", df.loc[:, ("col_A", "col_A.1")])  # Warning


# df.loc[:, ("col_B", "col_B.2", slice(None))] = 0
# df.loc[:, ("col_B", "col_B.2", slice(None))] = A  # No warning thrown
# df.loc[:, ("col_B", "col_B.2", slice(None))] = df[("col_A", "col_A.1")].sum(axis=1)  # No warning thrown
