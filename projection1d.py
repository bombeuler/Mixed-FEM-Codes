import numpy as np
from mpi4py import MPI

import gmsh
import pyvista as pv

from dolfinx import mesh, fem, plot
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import  assemble_matrix, assemble_vector
from dolfinx.io import gmshio

from petsc4py import PETSc

import ufl
from ufl import dx, inner, grad, div, ds
from ufl import TrialFunction, TestFunction, SpatialCoordinate

from untils.gmshDomain import lshape
from untils.getBoundary import getBoundary


class Projection1d:
    def __init__(self, domain, f, degree, fe_family):
        self.domain = domain
        self.V = functionspace(self.domain, (fe_family, degree))
        self.degree = degree
        self.fe_family = fe_family
        self.f = f

    def fromNSquare(n, f, degree, fe_family="Lagrange"):
        domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)
        return Projection1d(domain, f, degree, fe_family)

    @staticmethod
    def fromNRectangle(n, f, degree, fe_family="Lagrange"):
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            (np.array([-1, 0]), np.array([1, 1])),
            (2 * n, n),
            mesh.CellType.triangle,
        )
        return Projection1d(domain, f, degree, fe_family)

    @staticmethod
    def fromLshapeGmsh(n, f, degree, fe_family="Lagrange"):
        gmsh.initialize()
        if MPI.COMM_WORLD.rank == 0:
            model = lshape(gmsh, n)
        model = MPI.COMM_WORLD.bcast(model, root=0)
        domain, _, _ = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()
        MPI.COMM_WORLD.barrier()
        return Projection1d(domain, f, degree, fe_family)

    def setFemProblem(self):
        # ds = Measure("ds",domain=self.domain)
        x = SpatialCoordinate(self.domain)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        a = inner(u, v) * ds
        L = inner(self.f(x), v) * ds

        a_form = fem.form(a)
        l_form = fem.form(L)

        return (a_form, l_form)


    def solve(self):
        a_form, l_form = self.setFemProblem()
        dofs_bc = getBoundary(self.domain, self.V)
        dofs_all = np.arange(self.domain.geometry.x.shape[0], dtype=np.int32)
        dofs_inner = np.setdiff1d(dofs_all, dofs_bc)
        A_modify = np.ones_like(dofs_inner, dtype=np.floating)

        A = assemble_matrix(a_form)
        A.assemble()
        # A_inner = PETSc.Mat().createConstantDiagonal(dofs_inner.shape[0],1,self.domain.comm)
        # A_inner.assemble()
        A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        for i, values in enumerate(A_modify):
            A.setValue(dofs_inner[i], dofs_inner[i], values)
            if i % 1000 == 0:
                print(f"All:{dofs_inner.shape[0]},Already:{i}")
        A.assemble()

        b = assemble_vector(l_form)

        linear_solver = PETSc.KSP().create(self.domain.comm)
        linear_solver.setOperators(A)
        linear_solver.setType(PETSc.KSP.Type.PREONLY)
        linear_solver.getPC().setType(PETSc.PC.Type.LU)

        uh = fem.Function(self.V)
        uh.name = "uh"
        linear_solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        self.uh = uh
        self.dofs_inner = dofs_inner
        return uh

    def error(self):
        x = SpatialCoordinate(self.domain)
        ux = fem.Function(self.V)
        ux.interpolate(fem.Expression(self.f(x), self.V.element.interpolation_points()))
        ux.x.array[self.dofs_inner] = 0
        error_l2 = fem.form(inner(ux - self.f(x), ux - self.f(x)) * ds)
        error_l2_local = fem.assemble_scalar(error_l2)
        error_l2_global = self.domain.comm.allreduce(error_l2_local, op=MPI.SUM)
        error_l2 = np.sqrt(error_l2_global)
        return error_l2

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
            plotter.image_scale = 2
            plotter.save_graphic("ddd.eps")
            plotter.show()