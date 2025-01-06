import numpy as np
from mpi4py import MPI

import pyvista as pv

from dolfinx import mesh, fem, plot
from dolfinx import default_scalar_type
from dolfinx.fem import (
    functionspace,
    dirichletbc,
    create_nonmatching_meshes_interpolation_data,
)
from dolfinx.fem.petsc import LinearProblem

import ufl
from ufl import dx, inner, grad
from ufl import TrialFunction, TestFunction, SpatialCoordinate

import gmsh
from untils.gmshDomain import lshape
from dolfinx.io import gmshio


class PoissonSolver:
    # 对象实例化
    def __init__(self, domain, u_bc, degree, fe_family="Lagrange", u_exact=None):
        self.domain = domain
        self.V = functionspace(self.domain, (fe_family, degree))
        self.u_exact = u_exact
        self.degree = degree
        self.fe_family = fe_family
        self.f = fem.Constant(self.domain, default_scalar_type(0))
        self.u_bc = u_bc
        x = SpatialCoordinate(self.domain)

    # 指定n实例单位方形区域
    @staticmethod
    def fromNSquare(n, u_bc, degree, fe_family="Lagrange", u_exact=None):
        domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)
        return PoissonSolver(domain, u_bc, degree, fe_family, u_exact)

    @staticmethod
    def fromNRectangle(n, u_bc, degree, fe_family="Lagrange", u_exact=None):
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            (np.array([-1, 0]), np.array([1, 1])),
            (2 * n, n),
            mesh.CellType.triangle,
        )
        return PoissonSolver(domain, u_bc, degree, fe_family, u_exact)

    @staticmethod
    def fromLshapeGmsh(n, u_bc, degree, fe_family="Lagrange", u_exact=None):
        gmsh.initialize()
        if MPI.COMM_WORLD.rank == 0:
            model = lshape(gmsh, n)
        model = MPI.COMM_WORLD.bcast(model, root=0)
        domain, _, _ = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()
        MPI.COMM_WORLD.barrier()
        return PoissonSolver(domain, u_bc, degree, fe_family, u_exact)

    # 设置Dirichlet边界条件
    def setDirichletBC(self):
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        self.domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        x = SpatialCoordinate(self.domain)
        if isinstance(self.u_bc, ufl.Coefficient):
            nmm_data = create_nonmatching_meshes_interpolation_data(
                self.domain, self.V.element, self.u_bc.function_space.mesh
            )
            uD = fem.Function(self.V)
            uD.interpolate(self.u_bc, nmm_interpolation_data=nmm_data)
        else:
            uD = self.u_bc(x)
        dofs_1 = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
        bc_1 = dirichletbc(uD, dofs_1)  # type: ignore
        return [bc_1]

    # 设置变分问题
    def setFemProblem(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        a = inner(grad(u), grad(v)) * dx
        L = inner(self.f, v) * dx

        return (u, v, a, L)

    # 求解uh并且在对象中保存并返回uh
    def solve(self):

        bcs = self.setDirichletBC()
        u, v, a, L = self.setFemProblem()

        problem = LinearProblem(
            a,
            L,
            bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        # problem = LinearProblem(
        #     a,L,bcs,petsc_options={"ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000}
        # )
        uh = problem.solve()
        self.uh = uh
        return uh

    def outUh(self):
        return self.uh, self.V

    # 画uh的图像
    def plot(self):
        cells, types, x = plot.vtk_mesh(self.V)
        grid = pv.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = self.uh.x.array.real
        grid.set_active_scalars("u")
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        if pv.OFF_SCREEN:
            plotter.screenshot("uh_poisson.png")
        else:
            plotter.show_axes_all()
            plotter.show(auto_close=True)

    # 求解误差
    def error(self, degree_raise=3):
        if isinstance(self.u_exact, fem.Function):
            return self.__errorWithoutReal(degree_raise)
        else:
            return self.__errorWithReal(degree_raise)

    def __errorWithoutReal(self, degree_raise):
        ue_degree = self.u_exact.function_space.ufl_element().degree  # type: ignore
        ue_domain = self.u_exact.function_space.mesh  # type: ignore
        W = functionspace(ue_domain, (self.fe_family, ue_degree + degree_raise))

        comm = W.mesh.comm

        nmm_data = create_nonmatching_meshes_interpolation_data(
            W.mesh, W.element, self.domain
        )

        u_W = fem.Function(W)
        u_W.interpolate(self.uh, nmm_interpolation_data=nmm_data)  # type: ignore

        u_exact_W = fem.Function(W)
        u_exact_W.interpolate(self.u_exact(x))  # type: ignore

        # Compute the error in the higher order function space
        e_W = fem.Function(W)
        e_W.x.array[:] = u_W.x.array - u_exact_W.x.array  # type: ignore

        # Integrate the error
        error_l2 = fem.form(inner(e_W, e_W) * dx)
        error_l2_local = fem.assemble_scalar(error_l2)  # type: ignore
        error_l2_global = comm.allreduce(error_l2_local, op=MPI.SUM)
        self.error_l2 = np.sqrt(error_l2_global)

        error_h1 = fem.form(inner(e_W, e_W) * dx + inner(grad(e_W), grad(e_W)) * dx)
        error_h1_local = fem.assemble_scalar(error_h1)  # type: ignore
        error_h1_global = comm.allreduce(error_h1_local, op=MPI.SUM)
        self.error_h1 = np.sqrt(error_h1_global)

        error_max_local = np.max(np.abs(e_W.x.array))
        error_max_global = comm.allreduce(error_max_local, op=MPI.MAX)
        self.error_max = error_max_global

        return self.error_l2, self.error_h1, self.error_max

    def __errorWithReal(self, degree_raise):
        # x = SpatialCoordinate(self.domain)
        W = functionspace(self.domain, (self.fe_family, self.degree + degree_raise))
        comm = W.mesh.comm
        # Interpolate approximate solution
        u_W = fem.Function(W)
        u_W.interpolate(self.uh)  # type: ignore
        x = SpatialCoordinate(self.domain)
        u_exact_W = self.u_exact(x)
        u_exact_IW = fem.Function(self.V)
        u_exact_IW.interpolate(
            fem.Expression(u_exact_W, self.V.element.interpolation_points())
        )
        # Compute the error in the higher order function space

        error_l2 = fem.form(inner(u_W - u_exact_W, u_W - u_exact_W) * dx)
        error_l2_local = fem.assemble_scalar(error_l2)  # type: ignore
        error_l2_global = comm.allreduce(error_l2_local, op=MPI.SUM)
        self.error_l2 = np.sqrt(error_l2_global)

        error_h1 = fem.form(
            inner(u_W - u_exact_W, u_W - u_exact_W) * dx
            + inner(grad(u_W - u_exact_W), grad(u_W - u_exact_W)) * dx
        )
        error_h1_local = fem.assemble_scalar(error_h1)  # type: ignore
        error_h1_global = comm.allreduce(error_h1_local, op=MPI.SUM)
        self.error_h1 = np.sqrt(error_h1_global)

        error_max_local = np.max(np.abs(self.uh.x.array - u_exact_IW.x.array))
        error_max_global = comm.allreduce(error_max_local, op=MPI.MAX)
        self.error_max = error_max_global

        return self.error_l2, self.error_h1, self.error_max
