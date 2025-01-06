import ufl.conditional


from poissonMixedEOC import PoissonMixedEOC

if __name__ == "__main__":
    EPS = 0.0001
    filename_flag = f"lshape"

    theta = lambda x: ufl.conditional(
        ufl.ge(x[1], 0), ufl.atan2(x[1], x[0]), (ufl.atan2(x[1], x[0]) + 2 * ufl.pi)
    )
    r_x = lambda x: ufl.sqrt(abs(x[0] ** 2 + x[1] ** 2))
    uh_exact = lambda x: r_x(x) ** (-0.5 + EPS) * ufl.sin(
        (-0.5 + EPS) * theta(x)
    )

    eoc_lshape = PoissonMixedEOC(1,i_max=7,degree=1)
    eoc_lshape.computeError(uh_exact, uh_exact, 2)
    # eoc_square.computePlot()
    eoc_lshape.computeSave(uh_exact, uh_exact, 2,filename_flag=filename_flag)
