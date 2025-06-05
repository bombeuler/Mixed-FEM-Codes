import numpy as np
import pandas as pd
from revisionSolverMixed import RevisionSolverMixed
# from poissonSolver import PoissonSolver

from mpi4py import MPI

from basix.ufl import element
from basix import CellType
from projection1d import Projection1d

from ufl import dx, inner, grad,div
from dolfinx import fem

from ufl import SpatialCoordinate


class OtherDifferentMixedEOC:
    def __init__(
        self, domain_flag,i_max=5, degree=1, element1Family="Raviart-Thomas", element2Family="DG"
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
        compute_name="Poisson Equation Computing Errors with Mixed Types",
    ):
        print(compute_name)
        # solverPast = PoissonSolverMixed.from_BP(u_exact_filename, self.degree,self.element1Family,self.element2Family)
        for i in range(self.i_max):
            n = base ** (i + 1)

            if self.domain_flag == 0:
            # n = i+1
            # solver = PoissonSolverMixed.fromNSquare(n, self.degree,self.element1Family,self.element2Family,u_exact=u_exact)
                projection_pgn = Projection1d.fromNRectangle(n,gn, self.degree)
                pgn = projection_pgn.solve()
                # projection_pgn.plot()
                solver = RevisionSolverMixed.fromNRectangle(
                n,
                self.degree,
                self.element1Family,
                self.element2Family,
                gn=gn,
                pgn=pgn,
                u_exact=u_exact,
            )

            elif self.domain_flag == 1:
                projection_pgn = Projection1d.fromLshapeGmsh(n, gn, self.degree)
                pgn = projection_pgn.solve()
                solver = RevisionSolverMixed.fromLshapeGmsh(
                n,
                self.degree,
                self.element1Family,
                self.element2Family,
                gn=gn,
                pgn=pgn,
                u_exact=u_exact,
                )
                # solver.plotMesh()

            sigmah1,uh1,sigmah2,uh2 = solver.solve()
            # solver.plotPiecewise()
            # comm = uh.function_space.mesh.comm
            self.table_of_errors[i, 0] = 1 / n
            # self.table_of_errors[i, 1] = self.__error_real(uh2, u_exact)
            self.table_of_errors[i, 1] = self.__error(sigmah1, sigmah2)
            self.table_of_errors[i, 2] = self.__error(uh1, uh2)

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
        self.saveErrorsAsLatex(compute_name)

    def __error_real(self, sigma1, sigma2, degree_raise=3):
        domain = sigma1.function_space.mesh
        # elements = element(
        #     "RT",
        #     CellType.triangle,
        #     self.degree + degree_raise,
        #     shape=(2,),
        # )
        # W = fem.functionspace(domain, elements)
        comm = domain.comm
        x = SpatialCoordinate(domain)
        # u_W = fem.Function(W)
        # nmm_data1 = fem.create_nonmatching_meshes_interpolation_data(
        #     domain, W.element, sigma1.function_space.mesh
        # )

        # u_W.interpolate(sigma1, nmm_interpolation_data=nmm_data1)

        error_l2 = fem.form(inner(sigma1 - sigma2(x), sigma1 - sigma2(x)) * dx + inner(div(sigma1 - sigma2(x)), div(sigma1 - sigma2(x))) * dx)
        error_l2_local = fem.assemble_scalar(error_l2)
        error_l2_global = comm.allreduce(error_l2_local, op=MPI.SUM)
        error_l2 = np.sqrt(error_l2_global)
        return error_l2

    def __error(self, sigma1, sigma2, degree_raise=3):
        # domain = sigma2.function_space.mesh
        # elements = element(
        #     "RT",
        #     CellType.triangle,
        #     self.degree + degree_raise,
        #     shape=(2,),
        # )
        # W = fem.functionspace(domain, elements)
        comm = sigma1.function_space.mesh.comm
        # u_W = fem.Function(W)
        # nmm_data1 = fem.create_nonmatching_meshes_interpolation_data(
        #     domain, W.element, sigma1.function_space.mesh
        # )

        # u_W.interpolate(sigma1, nmm_interpolation_data=nmm_data1)

        error_l2 = fem.form(inner(sigma1 - sigma2, sigma1 - sigma2) * dx)
        error_l2_local = fem.assemble_scalar(error_l2)
        error_l2_global = comm.allreduce(error_l2_local, op=MPI.SUM)
        error_l2 = np.sqrt(error_l2_global)
        return error_l2

    def __makeErrorsTable(self, printed=True):
        table_columns = [
            "h",
            "sigma_h-tao_h:L2 Norm",
            "w_h-u_h:L2 Norm",
        ]
        table_columns2 = ["sigma_h-tao_h:L2 Rate","w_h-u_h:L2 Rate"]
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
            caption="计算误差",
            position="htb",
            column_format="cc",
        )
        rates_table.to_latex(
            f"Output/rates_{file_flag}.tex",
            index=False,
            caption="计算收敛阶",
            position="htb",
            column_format="c",
        )
