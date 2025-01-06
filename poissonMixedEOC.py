import numpy as np
import pandas as pd
from poissonSolverMixed import PoissonSolverMixed


class PoissonMixedEOC:
    def __init__(
        self,
        domain_flag,
        i_max=5,
        degree=1,
        element1Family="Raviart-Thomas",
        element2Family="DG",
    ):
        self.domain_flag = domain_flag
        self.i_max = i_max
        self.degree = degree
        self.table_of_errors = np.zeros((i_max, 3), dtype=np.float64)
        self.table_of_error_rates = np.zeros((i_max - 1, 2), dtype=np.float64)
        self.element1Family = element1Family
        self.element2Family = element2Family

    def computeError(
        self,
        u_exact,
        gn,
        base=2,
        compute_name="Computation of Poisson Equation Errors with Mixed Types",
    ):
        print(compute_name)
        # solverPast = PoissonSolverMixed.from_BP(u_exact_filename, self.degree,self.element1Family,self.element2Family)
        for i in range(self.i_max):
            n = base ** (i + 1)
            if self.domain_flag == 0:
                solver = PoissonSolverMixed.fromNRectangle(
                    n,
                    self.degree,
                    self.element1Family,
                    self.element2Family,
                    gn=gn,
                    u_exact=u_exact,
                )
            elif self.domain_flag == 1:
                solver = PoissonSolverMixed.fromLshapeGmsh(
                    n,
                    self.degree,
                    self.element1Family,
                    self.element2Family,
                    gn=gn,
                    u_exact=u_exact,
                )

            uh, sigmah, _ = solver.solve()
            # comm = uh.function_space.mesh.comm
            self.table_of_errors[i, 0] = 1 / n
            self.table_of_errors[i, 1:3] = solver.error("u")

        error_rates_primary = np.log(
            self.table_of_errors[1:, 1:] / self.table_of_errors[:-1, 1:]
        ).transpose()
        error_rates_divide = np.full(
            error_rates_primary.shape,
            np.log(self.table_of_errors[1:, 0] / self.table_of_errors[:-1, 0]),
        )
        self.table_of_error_rates = (
            error_rates_primary / error_rates_divide
        ).transpose()
        self.__makeErrorsTable()

    def __makeErrorsTable(self, printed=True):
        table_columns = [
            "h",
            "u:L2-Norm",
            "u:H1-Norm",
        ]
        table_columns2 = [
            "u:L2-Rate",
            "u:H1-Rate",
        ]
        errors_table = pd.DataFrame(self.table_of_errors, columns=table_columns)
        rates_table = pd.DataFrame(self.table_of_error_rates, columns=table_columns2)
        if printed:
            print(errors_table)
            print(rates_table)
        return errors_table, rates_table

    def saveErrorsAsLatex(self, file_flag=""):
        errors_table, rates_table = self.__makeErrorsTable(False)
        errors_table.to_latex(
            f"Output/errors_{file_flag}.tex",
            index=False,
            caption="Errors",
            position="htb",
            column_format="ccc",
        )
        rates_table.to_latex(
            f"Output/rates_{file_flag}.tex",
            index=False,
            caption="Rates",
            position="htb",
            column_format="cc",
        )

    def computePlot(self, u_exact, gn, base=2, compute_name="Poisson方程图像计算"):
        print(compute_name)
        for i in range(self.i_max):
            n = base ** (i + 1)
            if self.domain_flag == 0:
                solver = PoissonSolverMixed.fromNRectangle(
                    n,
                    self.degree,
                    self.element1Family,
                    self.element2Family,
                    gn=gn,
                    u_exact=u_exact,
                )
            elif self.domain_flag == 1:
                solver = PoissonSolverMixed.fromLshapeGmsh(
                    n,
                    self.degree,
                    self.element1Family,
                    self.element2Family,
                    gn=gn,
                    u_exact=u_exact,
                )
            solver.solve()
            # solver.plot_mesh(f"{n}_lshape")
            # solver.plotPiecewise(f"{n}_square")
            # solver.plot(f"{n}_lshape")
            # solver.plot_mesh(f"{n}_square")
            # solver.plot(f"{n}_square")

    def computeSave(
        self,
        u_exact,
        gn,
        base=2,
        compute_name="Output of Computation Data ",
        filename_flag="",
    ):
        print(compute_name)
        for i in range(self.i_max):
            n = base ** (i + 1)
            if self.domain_flag == 0:
                solver = PoissonSolverMixed.fromNRectangle(
                n,
                self.degree,
                self.element1Family,
                self.element2Family,
                gn=gn,
                u_exact=u_exact,
            )
            elif self.domain_flag == 1:
                solver = PoissonSolverMixed.fromLshapeGmsh(
                n,
                self.degree,
                self.element1Family,
                self.element2Family,
                gn=gn,
                u_exact=u_exact,
            )
            solver.solve()

            solver.saveMat(f"{n}_domain={self.domain_flag}_{filename_flag}")
            print(f"第{i}轮计算结果已保存")
