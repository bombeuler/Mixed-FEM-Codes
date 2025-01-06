import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from untils.parseFemData import parseFemData

def plotPiecewiseFunc(uh, plotter):
 
    cells,points,u_values = parseFemData(uh)


    # scipy.io.savemat("./Storage/uh.mat",{'cells':cells_var,'points':points,'values':u_values})

    # 创建 Matplotlib colormap
    cmap = plt.get_cmap("jet")

    mins = u_values.min()
    maxs = u_values.max()
    d = maxs - mins
    print(f"mins:{mins},maxs:{maxs},d:{d}")
    for ii in range(len(cells)):
        cell = cells.links(ii)
        # value =uh.x.array[ii]
        value = u_values[ii]
        # print(f"value:{value}")
        local_points = np.concatenate(
            (points[cell][:, :-1], np.repeat(value, 3).reshape(-1, 1)), axis=1
        )
        faces = np.array([[3, 0, 1, 2]])
        surf = pv.PolyData(local_points, faces)

        # 获取颜色映射
        color = cmap((value - mins) / d)[:3]  # 归一化并获取 RGB 颜色
        plotter.add_mesh(surf, show_edges=True, color=color)
    zeros = pv.PolyData(np.array([[-1, -1, 0],[1,-1,0],[1,1,0],[-1,1,0]]), np.array([[4, 0, 1, 2,3]]))
    plotter.add_mesh(zeros, show_edges=True, color="blue",opacity=0.3)
