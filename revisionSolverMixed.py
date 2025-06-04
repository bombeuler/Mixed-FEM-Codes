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

from dolfinx.io import gmshio

import ufl
from ufl import dx, ds, inner, grad, div
from ufl import TrialFunctions, TestFunctions, SpatialCoordinate, FacetNormal

from basix.ufl import element, mixed_element

import ufl.conditional
import gmsh

from untils.plotPiecewiseFunc import plotPiecewiseFunc

from untils.gmshDomain import lshape,lshape_nonsymm

from untils.pyvistaParameter import setPlotterParameter
from untils.parseFemData import parseFemData


class RevisionSolverMixed:
    # 对象实例化
    def __init__(
        self, domain, degree, element1Family, element2Family, gn=None,pgn=None, u_exact=None
    ):
        self.domain = domain
        x = SpatialCoordinate(self.domain)
        self.element1Family = element1Family
        self.element2Family = element2Family
        element1 = element(
            element1Family,
            domain.basix_cell(),
            degree,
            shape=(self.domain.topology.dim,),
        )
        element2 = element(element2Family, domain.basix_cell(), degree - 1)
        self.mixelement = mixed_element([element1, element2])
        self.V = functionspace(domain, self.mixelement)
        self.degree = degree
        self.u_exact = u_exact
        self.f = fem.Constant(self.domain, default_scalar_type(0))
        self.gn = gn
        self.pgn = pgn

    @staticmethod
    def fromNCube(n, degree, element1Family, element2Family, gn=None,pgn=None, u_exact=None):
        domain = mesh.create_unit_cube(
            MPI.COMM_WORLD, n, n, n, mesh.CellType.tetrahedron
        )
        return RevisionSolverMixed(
            domain, degree, element1Family, element2Family, gn,pgn, u_exact
        )

    # 指定n实例单位方形区域
    @staticmethod
    def fromNSquare(n, degree, element1Family, element2Family, gn=None,pgn=None, u_exact=None):
        domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)
        return RevisionSolverMixed(
            domain, degree, element1Family, element2Family, gn,pgn, u_exact
        )

    @staticmethod
    def fromNRectangle(
        n, degree, element1Family, element2Family, gn=None, pgn=None,u_exact=None
    ):
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            (np.array([-1, 0]), np.array([1, 1])),
            (2 * n, n),
            mesh.CellType.triangle,
        )
        return RevisionSolverMixed(
            domain, degree, element1Family, element2Family, gn,pgn, u_exact
        )

    @staticmethod
    def fromLshapeGmsh(
        n, degree, element1Family, element2Family, gn=None,pgn=None, u_exact=None
    ):
        gmsh.initialize()
        if MPI.COMM_WORLD.rank == 0:
            model = lshape_nonsymm(gmsh, n)
        model = MPI.COMM_WORLD.bcast(model, root=0)
        domain, _, _ = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()
        MPI.COMM_WORLD.barrier()
        return RevisionSolverMixed(
            domain, degree, element1Family, element2Family, gn,pgn, u_exact
        )


    # 设置Dirichlet边界条件
    def setDirichletBC(self):

        V0 = self.V.sub(0)
        fdim = self.domain.topology.dim - 1
        facets_top = mesh.locate_entities_boundary(
            self.domain,
            fdim,
            lambda x: np.logical_and(
                np.isclose(x[0], 0.0),
                np.logical_and(np.greater(x[1], 0), np.less(x[1], 1)),
            ),
        )
        Q, _ = V0.collapse()
        dofs_1 = fem.locate_dofs_topological((V0, Q), fdim, facets_top)
        uD = fem.Function(Q)
        uD.interpolate(lambda x: np.zeros((2, x.shape[1])))
        bc_1 = dirichletbc(uD, dofs_1, V0)

        bcs = [bc_1]
        return bcs

        # tdim = self.domain.topology.dim
        # fdim = tdim - 1
        # self.domain.topology.create_connectivity(fdim, tdim)
        # boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        # uD = ScalarType(0.)
        # # uD = fem.Function(self.V)
        # # uD.interpolate(lambda x: x[0] ** 3 - 3 * x[0] * x[1] ** 2 - x[0] ** 2 / 2)
        # dofs_1 = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
        # bc_1 = dirichletbc(uD, dofs_1,V)
        # return [bc_1]

    # 设置变分问题
    def setFemProblem(self):
        sigma, u = TrialFunctions(self.V)
        tau, v = TestFunctions(self.V)
        x = SpatialCoordinate(self.domain)
        n = FacetNormal(self.domain)

        a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
        if isinstance(self.gn, ufl.Coefficient):
            # W, _ = self.V.sub(1).collapse()
            W2 = functionspace(self.domain,("Lagrange", self.degree))
            nmm_data = create_nonmatching_meshes_interpolation_data(
                self.domain, W2.element, self.gn.function_space.mesh
            )
            gn = fem.Function(W2)
            gn.interpolate(self.gn, nmm_interpolation_data=nmm_data)
        else:
            gn = self.gn(x)

        # W, _ = self.V.sub(1).collapse()
        W2 = functionspace(self.domain,("Lagrange", self.degree))
        # W2 = functionspace(self.domain, ("DG", self.degree - 1))
        nmm_data = create_nonmatching_meshes_interpolation_data(
                self.domain, W2.element, self.pgn.function_space.mesh
            )
        pgn = fem.Function(W2)
        pgn.interpolate(self.pgn, nmm_interpolation_data=nmm_data)

        L1 = -inner(self.f, v) * dx + gn * inner(tau, n) * ds
        L2 = -inner(self.f, v) * dx + pgn * inner(tau, n) * ds
        return (a, L1,L2)

    # 求解uh并且在对象中保存并返回uh
    def solve(self):

        # bcs = self.setDirichletBC()
        a, L1,L2 = self.setFemProblem()

        problem1 = LinearProblem(
            a,
            L1,
            # bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        problem2 = LinearProblem(
            a,
            L2,
            # bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        wh1 = problem1.solve()
        wh2 = problem2.solve()
        sigmah1, uh1 = wh1.split()
        sigmah2, uh2 = wh2.split()
        self.sigmah1,self.uh1, self.sigmah2,self.uh2 = sigmah1.collapse(),uh1.collapse(),sigmah2.collapse(),uh2.collapse()
        return sigmah1.collapse(),uh1.collapse(),sigmah2.collapse(),uh2.collapse()

    def saveMat(self, filename):
        if self.uh is None:
            raise ReferenceError("必顿先解出uh才能保存数据")
        else:
            path = f"./Storage/{filename}.mat"
            parseFemData(self.uh.collapse(), path,saved=True)

    def saveError(self,filename):
        if self.uh is None:
            raise ReferenceError("必顿先解出uh才能保存数据")
        else:
            path = f"./Storage/{filename}_errors.mat"
            uhc = self.uh.collapse()
            W = uhc.function_space
            x = SpatialCoordinate(W.mesh)
            eh = fem.Function(W)
            uex = fem.Function(W)
            uex.interpolate(fem.Expression(self.u_exact(x),W.element.interpolation_points()))
            eh.x.array[:] = uhc.x.array - uex.x.array
            
            parseFemData(eh, path,saved=True)

    def plotPiecewise(self, filename=None):
        plotter = pv.Plotter(window_size=(1200, 1200))
        plotPiecewiseFunc(self.uh2, plotter)
        # plotter.view_xy()
        # plotter.add_mesh(grid, show_edges=True)
        # setPlotterParameter(plotter, "bird-xy")
        plotter.set_scale(zscale=0.25)
        if pv.OFF_SCREEN:
            plotter.screenshot("uh_poisson.png")
        else:
            plotter.show_axes_all()
            plotter.show()

    def plot(self, filename=None):

        VuP = functionspace(self.domain, ("CG", 1))
        VsP = functionspace(self.domain, ("CG", 1, (2,)))
        uhP = fem.Function(VuP)
        uhP.interpolate(self.uh2)
        sigmahP = fem.Function(VsP)
        sigmahP.interpolate(self.sigmah2)

        plotter = pv.Plotter(window_size=(1200, 1200))

        # plotter.subplot(0, 0)
        cells, types, geometry = plot.vtk_mesh(VuP)
        grid = pv.UnstructuredGrid(cells, types, geometry)
        grid.point_data["u"] = uhP.x.array.real
        grid.set_active_scalars("u")
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        # setPlotterParameter(plotter, "bird-xy")

        if pv.OFF_SCREEN:
            plotter.screenshot("uh_poisson.png")
        else:
            plotter.show_axes_all()
            plotter.show()

    def plotMesh(self, filename=None):

        VuP = functionspace(self.domain, ("CG", 1))
        # VsP = functionspace(self.domain, ("CG", 1, (2,)))
        # uhP = fem.Function(VuP)
        # uhP.interpolate(self.uh)
        # sigmahP = fem.Function(VsP)
        # sigmahP.interpolate(self.sigmah)

        plotter = pv.Plotter(window_size=(1200, 1200))

        cells, types, geometry = plot.vtk_mesh(VuP)
        grid = pv.UnstructuredGrid(cells, types, geometry)
        plotter.add_mesh(grid, style="wireframe", line_width=8, color="k")
        plotter.view_xy()
        plotter.add_axes(zlabel="", z_color=[0, 0, 0, 0])
        plotter.show()

    # 求解误差
    def error(self, name, degree_raise=1):
        if isinstance(self.u_exact, fem.Function) or isinstance(self.u_exact, tuple):
            return self.__errorWithoutReal(name, degree_raise)
        else:
            return self.__errorWithReal(name, degree_raise)

    def outSigma(self):
        W, _ = self.V.sub(0).collapse()
        return self.sigmah, W

    def outU(self):
        W, _ = self.V.sub(1).collapse()

        return self.uh, W

    def __errorWithoutReal(self, name, degree_raise):
        sigma_exact, u_exact = self.u_exact.split()
        if name == "u":
            ue_degree = u_exact.function_space.ufl_element().degree
            ue_domain = u_exact.function_space.mesh
            elements = element(
                self.element2Family, ue_domain.basix_cell(), ue_degree + degree_raise
            )

            W = functionspace(self.domain, elements)

            comm = W.mesh.comm
            nmm_data = create_nonmatching_meshes_interpolation_data(
                W.mesh, W.element, ue_domain
            )
            u_W = fem.Function(W)
            u_W.interpolate(self.uh)
            # u_W.interpolate(self.uh,nmm_interpolation_data=nmm_data)
            u_exact_W = fem.Function(W)
            u_exact_W.interpolate(u_exact, nmm_interpolation_data=nmm_data)

            e_W = fem.Function(W)
            e_W.x.array[:] = u_W.x.array - u_exact_W.x.array

            error_l2 = fem.form(inner(e_W, e_W) * dx)
            error_l2_local = fem.assemble_scalar(error_l2)
            error_l2_global = comm.allreduce(error_l2_local, op=MPI.SUM)
            error_l2 = np.sqrt(error_l2_global)

            error_h1 = fem.form(inner(e_W, e_W) * dx + inner(grad(e_W),grad(e_W)) * dx)
            error_h1_local = fem.assemble_scalar(error_h1)
            error_h1_global = comm.allreduce(error_h1_local, op=MPI.SUM)
            error_h1 = np.sqrt(error_h1_global)

            return self.error_l2, self.error_h1


    def __errorWithReal(self, name, degree_raise):
        x = SpatialCoordinate(self.domain)
        cutoff = 1
        if name == "u":
            elements = element(
                self.element2Family,
                self.domain.basix_cell(),
                self.degree + degree_raise,
            )
            W = functionspace(self.domain, elements)
            comm = W.mesh.comm
            u_W = fem.Function(W)
            u_W.interpolate(self.uh)


            u_exact_W = self.u_exact(x)
            error_l2 = fem.form(cutoff * inner(u_W - u_exact_W, u_W - u_exact_W) * dx)
            error_l2_local = fem.assemble_scalar(error_l2)
            error_l2_global = comm.allreduce(error_l2_local, op=MPI.SUM)
            error_l2 = np.sqrt(error_l2_global)

            error_h1 = fem.form(
                cutoff * inner(u_W - u_exact_W, u_W - u_exact_W) * dx
                + cutoff * inner(grad(u_W - u_exact_W), grad(u_W - u_exact_W)) * dx
                # + inner(div(grad(u_W - u_exact_W)), div(grad(u_W - u_exact_W))) * dx
            )
            error_h1_local = fem.assemble_scalar(error_h1)
            error_h1_global = comm.allreduce(error_h1_local, op=MPI.SUM)
            error_h1 = np.sqrt(error_h1_global)
        return error_l2, error_h1
