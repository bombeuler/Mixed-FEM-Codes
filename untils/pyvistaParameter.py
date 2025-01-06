def setPlotterParameter(plotter,name="bird-xy"):
    if name == "bird-xy":
        parameter_bird_xy(plotter)

def parameter_bird_xy(plotter):
    plotter.camera.position = (0, -6, -1)
    plotter.camera.focal_point = (0, 0, -1)
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.elevation = 45
    plotter.camera.azimuth = 30