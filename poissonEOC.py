import numpy as np
from poissonSolver import PoissonSolver
from dolfinx import default_scalar_type


class PoissonEOC:
    def __init__(self, i_max=5, degree=1):
        self.i_max = i_max
        self.degree = degree
        self.l2_errors = np.zeros(i_max, dtype=default_scalar_type)
        self.h1_errors = np.zeros(i_max, dtype=default_scalar_type)
        self.max_errors = np.zeros(i_max, dtype=default_scalar_type)
        self.hs = np.zeros(i_max, dtype=np.float64)

    def computeError(self, u_exact, gn, base=3, compute_name="Computation of Poisson Problem Errors"):
        print(compute_name)
        # solverPast = PoissonSolver.from_BP(u_exact_filename, self.degree)
        print(f"h:    Mesh     Error:  L2 Norm    ---   H1 Norm   ---   Max Norm")
        for i in range(self.i_max):
            n = base ** (i + 1)
            # solver = PoissonSolver.fromNSquare(n, self.degree,u_exact=solverPast.uh)
            solver = PoissonSolver.fromNRectangle(n, self.degree, u_exact=u_exact)
            uh = solver.solve()
            comm = uh.function_space.mesh.comm
            self.l2_errors[i], self.h1_errors[i], self.max_errors[i] = solver.error()
            self.hs[i] = 1 / n
            if comm.rank == 0:
                print(
                    f"h: {self.hs[i]:.5e} Error: {self.l2_errors[i]:.5e} --- {self.h1_errors[i]:.5e} --- {self.max_errors[i]:.5e}"
                )
        rates_l2 = np.log(self.l2_errors[1:] / self.l2_errors[:-1]) / np.log(
            self.hs[1:] / self.hs[:-1]
        )
        rates_h1 = np.log(self.h1_errors[1:] / self.h1_errors[:-1]) / np.log(
            self.hs[1:] / self.hs[:-1]
        )
        rates_max = np.log(self.max_errors[1:] / self.max_errors[:-1]) / np.log(
            self.hs[1:] / self.hs[:-1]
        )
        if comm.rank == 0:
            print(f"Polynomial degree {self.degree:d}, L2 Norm Rates {rates_l2}")
            print(f"Polynomial degree {self.degree:d}, H1 Norm Rates {rates_h1}")
            print(f"Polynomial degree {self.degree:d}, Max Norm Rates {rates_max}")

    def computePlot(self, compute_name="Plots of Poisson Problem"):
        print(compute_name)
        for i in range(self.i_max):
            n = 2 ** (i + 1)
            # solver = PoissonSolver.fromNSquare(n, self.degree)
            solver = PoissonSolver.fromNRectangle(n, self.degree)
            solver.solve()
            solver.plot()
