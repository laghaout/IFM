    def decompose_linear(self, delimiter=DELIMITER):

        # TODO: Replace self.report by self.self.report
        # Reshape the matrices as vectors
        matrix = self.report.actual.rho.map(lambda x: x.reshape(-1, 1))
        decompositions = matrix.columns[1:]
        matrix["Vecs"] = matrix.apply(
            lambda x: np.hstack([x[k] for k in decompositions]), axis=1
        )
        matrix.drop(decompositions, axis=1, inplace=True)
        # matrix['res'] = matrix.apply(lambda x: x[np.nan], axis=1)
        matrix["res"] = matrix.apply(
            lambda x: np.linalg.lstsq(x["Vecs"], x[np.nan].reshape(-1, 1), rcond=None),
            axis=1,
        )

        for i, col in enumerate(self.report.linear.weight.columns):
            self.report.loc[:, ("linear", "weight", col)] = matrix["res"].apply(
                lambda x: qi.trim_imaginary(x[0][i][0])
            )  # x[0][c]

        self.report.loc[:, ("linear", "residuals", None)] = matrix["res"].apply(
            lambda x: x[0][1]
        )

        # self.report.loc[:, ("linear", "rho", slice(None))] = self.report.apply(
        #     lambda x: (x[("actual", "rho")][1:] @ x[("linear", "weight")]), axis=1
        # )

        self.report.loc[:, ("linear", "rho", slice(None))] = self.report.apply(
            lambda x: (x[("actual", "rho")][1:] @ x[("linear", "weight")]), axis=1
        )

        self.report.loc[:, ("linear", "rho", slice(None))] = self.report.loc[
            :, ("linear", "rho", slice(None))
        ].apply(lambda x: x.values.item(), axis=1)

        # A = self.report.loc[:, ("linear", "rho", np.nan)].iloc[3]

        self.report[("linear", "purity", None)] = self.report[("linear", "rho", None)].apply(
            lambda x: qi.purity(x) if qi.is_density_matrix(x) else np.nan
        )
        self.report[("linear", "fidelity", None)] = self.report.apply(
            lambda x: qi.fidelity(
                x[("actual", "rho", None)], x[("linear", "rho", None)]
            )
            if qi.is_density_matrix(x[("linear", "rho", None)])
            else np.nan,
            axis=1,
        )
