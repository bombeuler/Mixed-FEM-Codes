import numpy as np
import scipy as sp


def parseFemData(uh,path="",saved = False):
    domain = uh.function_space.mesh
    dim = domain.topology.dim
    cells = domain.topology.connectivity(dim, 0)
    points = domain.geometry.x
    u_values = uh.x.array.real
    cells_var = np.zeros(shape=(len(cells),3),dtype=np.int64)
    for ii in range(len(cells)):
        cell = cells.links(ii)
        cells_var[ii,:] = cell

    if saved:
        sp.io.savemat(path,{'cells':cells_var,'points':points,'values':u_values})

    return cells_var,points,u_values
