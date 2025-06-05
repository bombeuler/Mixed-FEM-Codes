import ufl.conditional


from otherDifferentMixed import OtherDifferentMixedEOC

if __name__ == "__main__":
    EPS = 0.4
    filename_flag = f"revision_lshape_test"

    theta = lambda x: ufl.conditional(
        ufl.ge(x[1], 0), ufl.atan2(x[1], x[0]), (ufl.atan2(x[1], x[0]) + 2 * ufl.pi)
    )
    r_x = lambda x: ufl.sqrt(abs(x[0] ** 2 + x[1] ** 2))
    uh_exact = lambda x: r_x(x) ** (-0.5 + EPS) * ufl.sin((-0.5 + EPS) * theta(x))
    # uh_exact = lambda x: ufl.sin(x[0]) * ufl.sin(x[1])
    # uh_exact  = lambda x: x[0]
    eoc_square = OtherDifferentMixedEOC(0, i_max=7, degree=1)
    eoc_square.computeError(uh_exact, uh_exact)
    eoc_square.saveErrorsAsLatex(filename_flag)
