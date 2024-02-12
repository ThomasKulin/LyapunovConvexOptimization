import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from pydrake.all import (
    RegionOfAttraction,
    RegionOfAttractionOptions,
    SymbolicVectorSystem,
    Variable,
    plot_sublevelset_expression,
)

from underactuated import plot_2d_phase_portrait, running_as_notebook

def star_convex():
    # Construct a non-convex 2D level set.
    x = np.array([Variable("x"), Variable("y")]).reshape((2,))
    A1 = np.array([[1, 2], [3, 4]])
    A2 = A1 @ np.array([[-1, 0], [0, 1]])  # mirror about y-axis
    U = (x.T.dot(A1.T.dot(A1.dot(x)))) * (x.T.dot(A2.T.dot(A2.dot(x))))

    fig, ax = plt.subplots()

    theta = 0.5
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    dUdx = U.Jacobian(x)
    sys = SymbolicVectorSystem(state=x, dynamics=(U - 1) * dUdx.T)
    context = sys.CreateDefaultContext()

    options = RegionOfAttractionOptions()
    options.lyapunov_candidate = x.dot(x)
    options.state_variables = x
    V = RegionOfAttraction(sys, context, options)
    plot_sublevelset_expression(ax, V)
    plot_sublevelset_expression(ax, U, 101, linewidth=3, fill=False)

    plot_2d_phase_portrait(sys,(-0.8, 0.8), (-0.6, 0.6))

star_convex()
