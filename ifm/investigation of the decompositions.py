def foo_Born(x):
    combis = system.combis.index[1:]
    weights = report.born.weight[combis].loc[x.name]
    purities = x[combis].to_list()

    return purities @ weights


def foo_p58(x):
    from math import comb

    N = system.N
    C = comb(N, 2)

    disturbed = [j for j in x.index if str(x.name[1]) in j and DELIMITER in j]
    disturbed = np.sum([x[k] for k in disturbed])

    undisturbed = [
        j for j in x.index if str(x.name[1]) not in j and DELIMITER in j
    ]
    undisturbed = np.sum([x[k] for k in undisturbed])
    # undisturbed = (C - N + 1)*x['initial']

    return (disturbed + undisturbed) / C


def foo(N, u, d):
    from math import comb

    C = comb(N, 2)
    return (C * d + (N - 1 - C) * u) / (N - 1)


A = report.actual.purity
A["Born"] = A.apply(foo_Born, axis=1)
A["p58"] = A.apply(foo_p58, axis=1)
print(A[["final", "p58", "Born"]])

from scipy.optimize import minimize

X = A[system.combis.index[1:]]  # .iloc[:17]
y = A["final"]


def objective(x):
    return np.sum((X.dot(x) - y) ** 2)


# Constraints
cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Sum to unity
bounds = [(0, 1)] * X.shape[1]  # Probability constraints for each element of x

# Initial guess
x0 = np.random.rand(X.shape[1])
x0 /= np.sum(x0)  # Normalize to satisfy the sum to unity constraint initially

# Solve the constrained optimization problem
result = minimize(objective, x0, bounds=bounds, constraints=cons)

if result.success:
    print("Optimal solution found:", result.x)
else:
    print("Optimization failed.")


def foo_convex(x):
    combis = system.combis.index[1:]
    weights = result.x
    purities = x[combis].to_list()

    return purities @ weights


A["convex"] = A.apply(foo_convex, axis=1)

print(A[["final", "convex", "p58", "Born"]])

if False:

    def hash_matrix(matrix, encoding="utf-8", sha_round=6):
        import hashlib

        matrix = str(np.round(matrix.reshape(-1, 1), sha_round))

        # Convert the string to bytes
        input_bytes = matrix.encode(encoding)

        # Create a sha256 hash object
        hash_object = hashlib.sha256(input_bytes)

        # Generate the hexadecimal representation of the SHA hash
        sha_value = hash_object.hexdigest()

        return sha_value[-sha_round:]

    actual = report.actual.rho.copy().T

    undisturbed = pd.DataFrame([actual.columns], columns=actual.columns)
    undisturbed.rename(index={0: "undisturbed"}, inplace=True)
    undisturbed = undisturbed.map(
        lambda x: np.outer(system.b[x[1] - 1][1:], system.b[x[1] - 1][1:])
    )
    actual = pd.concat([undisturbed, actual], axis=0).T
    actual = actual.map(lambda x: x.astype("complex"))

    A = list()
    for k in range(len(report)):
        # A.append(actual.iloc[k])
        A.append(actual.map(hash_matrix).iloc[k])
    A = pd.concat(A, axis=1).T
    # A = pd.concat(
    #     [report.actual.rho.iloc[0], actual.iloc[0],
    #      report.actual.rho.iloc[1], actual.iloc[1]], axis=1)

    # A['unique'] = A.apply(lambda x: x.unique(), axis=1)
    # A['cardinality'] = A['unique'].map(lambda x: len(x))

    letters = [chr(i) for i in range(65, 91)]
    letters = {
        v: (f"D{k}" if v not in A.undisturbed.loc[1].unique().tolist() else k)
        for k, v in enumerate(set(A.melt().value))
    }
    A = A.map(lambda x: letters[x])

elif False:

    def check_with_old(
        report=report,
        reconstruction_linear=reconstruction_linear,
        decomposed_rho=decomposed_rho,
    ):
        for o, b in report.index:
            print(
                np.allclose(
                    report.loc[(o, b), ("linear", "rho")].values.item(),
                    reconstruction_linear[o][b],
                ),
                "linear reconstruction",
            )

            print(
                np.allclose(
                    report.loc[(o, b), ("actual", "rho", None)],
                    output_rho[o][b],
                ),
                "actual",
            )

            decomp = "1" + DELIMITER + "3"
            print(
                np.allclose(
                    report.loc[(o, b), ("actual", "rho", decomp)],
                    decomposed_rho[decomp][o][b],
                ),
                "decomposed_rho",
            )
            # print(
            #     (
            #         report.loc[(o, b), ("linear", "rho")].values.item()
            #         - reconstruction_linear[o][b]
            #     ).sum()
            # )
            # print(
            #     (
            #         report.loc[(o, b), ("linear", "rho")].values.item()
            #         - report.loc[(o, b), ("linear", "rho")].values.item()
            #     ).sum()
            # )

    check_with_old()
