from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import exterior_facet_indices

def getBoundary(domain,V):
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = exterior_facet_indices(domain.topology)
    dofs = locate_dofs_topological(V, fdim, boundary_facets)

    return dofs