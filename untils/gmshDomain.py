import numpy as np

def lshape(gmsh, n_nodes=1):
    gmsh.option.setNumber("General.Terminal", 0)
    factory = gmsh.model.geo
    lc = 1.
    p1 = factory.addPoint(-1., -1., 0., lc)
    p2 = factory.addPoint(0., -1., 0., lc)
    p3 = factory.addPoint(0., 0., 0., lc)
    p4 = factory.addPoint(1., 0., 0., lc)
    p5 = factory.addPoint(1., 1., 0., lc)
    p6 = factory.addPoint(0., 1., 0., lc)
    p7 = factory.addPoint(-1., 1., 0., lc)
    p8 = factory.addPoint(-1., 0., 0., lc)

    l1 = factory.addLine(p1,p2)
    l2 = factory.addLine(p2,p3)
    l3 = factory.addLine(p3,p4)
    l4 = factory.addLine(p4,p5)
    l5 = factory.addLine(p5,p6)
    l6 = factory.addLine(p6,p7)
    l7 = factory.addLine(p7,p8)
    l8 = factory.addLine(p8,p1)


    cl1 = factory.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8])

    s1 = factory.addPlaneSurface([cl1])

    # dx = 5.
    # num_els_z = 10
    # factory.extrude([(2, s1)], 0., 0., dx,
    #                     numElements=[2*num_els_z-1], recombine=True)

    factory.synchronize()
    gmsh.model.addPhysicalGroup(2,[s1],1,name="l shape")

    meshFact = gmsh.model.mesh

    # gmsh.option.setNumber("Mesh.Smoothing", 100)
    # gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
    # gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    meshFact.setAlgorithm(2,1,3)
    meshFact.generate(2)
    loop_flag = np.log2(n_nodes)
    while(loop_flag > 0):
        meshFact.refine()
        loop_flag -=1

    # factory = gmsh.model.occ

    # factory.addRectangle(-1,-1,0,1,2,1)
    # factory.addRectangle(0,0,0,1,1,2)
    # # factory.fuse([(2,1)],[(2,2)],3)
    # factory.synchronize()

    # meshFact = gmsh.model.mesh
    # n_nodes +=1
    # n_nodes2 = 2*n_nodes -1 

    # boundarys1 = gmsh.model.getBoundary([(2,1)])
    # # print(boundarys1)
    # for i,boundary in boundarys1:
    #     if boundary % 2 == 0:
    #         print(boundary)
    #         meshFact.setTransfiniteCurve(boundary, numNodes=n_nodes2)
    #     else:
    #         meshFact.setTransfiniteCurve(boundary, numNodes=n_nodes)

    # boundarys2 = gmsh.model.getBoundary([(2,2)])
    # # print(boundarys2)
    # for i,boundary in boundarys2:
    #     meshFact.setTransfiniteCurve(boundary, numNodes=n_nodes)

    # meshFact.setTransfiniteSurface(1)
    # meshFact.setTransfiniteSurface(2)
    # # factory.fuse([(2,1)],[(2,2)],3,removeObject=False,removeTool=False)
    # gmsh.model.addPhysicalGroup(2,[1,2],1)
    # # gmsh.model.addPhysicalGroup(1,[1,2,3,4,5,6,7],1)
    # meshFact.generate(2)
    # # factory.cut([(1,2)],[(1,8)],9)
    # # gmsh.fltk.run()

    return gmsh.model

def lshape_boundary(gmsh,num_points,size=1):
    num_points += 1

    gmsh.model.add("1D L-Shape Mesh")
    gmsh.option.setNumber("General.Terminal", 0)

    # Define the points of the L-shape
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(size, 0, 0)
    p3 = gmsh.model.geo.addPoint(size, size, 0)
    p4 = gmsh.model.geo.addPoint(0, size, 0)
    p5 = gmsh.model.geo.addPoint(-size, size, 0)
    p6 = gmsh.model.geo.addPoint(-size, 0, 0)
    p7 = gmsh.model.geo.addPoint(-size, -size, 0)
    p8 = gmsh.model.geo.addPoint(0, -size, 0)

    # Define the lines of the L-shape
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p1)

    # Create a loop and a surface
    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7,l8])
    gmsh.model.geo.addPlaneSurface([cl1])

    # Synchronize the model
    gmsh.model.geo.synchronize()

    # Define the mesh size
    gmsh.model.mesh.setTransfiniteCurve(l1, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l2, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l3, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l4, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l5, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l6, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l7, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l8, num_points)

    # Add physical groups for the lines
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4, l5, l6, l7, l8], 1)
    gmsh.model.addPhysicalGroup(1,[cl1],2)

    # Generate the 1D mesh
    gmsh.model.mesh.generate(1)

    return gmsh.model

def square_boundary(gmsh, num_points, size=1):
    num_points += 1

    gmsh.model.add("1D Square Mesh")
    gmsh.option.setNumber("General.Terminal", 0)

    # Define the points of the square
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(size, 0, 0)
    p3 = gmsh.model.geo.addPoint(size, size, 0)
    p4 = gmsh.model.geo.addPoint(0, size, 0)

    # Define the lines of the square
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create a loop and a surface
    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    # gmsh.model.geo.addPlaneSurface([cl1])

    # Synchronize the model
    gmsh.model.geo.synchronize()

    # Define the mesh size
    gmsh.model.mesh.setTransfiniteCurve(l1, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l2, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l3, num_points)
    gmsh.model.mesh.setTransfiniteCurve(l4, num_points)

    # Add physical groups for the lines
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], 1)
    # gmsh.model.addPhysicalGroup(1, [cl1], 2)
    # Generate the 1D mesh
    gmsh.model.mesh.generate(1)

    return gmsh.model