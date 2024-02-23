import plotly.graph_objects as go
import plotly.plotly as py
import plotly.figure_factory as FF

import numpy as np
from scipy.spatial import Delaunay
from util import samples

def do_plotting(CL, sim=True):
    if sim:
        final = simOneTraj(CL, 100, [init_x_train, init_c], num_samples)
        finalx = final[0]
        # TODO: extract plant name from the CL
        np.save('../data/double_sim_100steps.npy', finalx)
    else:
        finalx = np.load('../data/double_sim.npy')

    # final_theta = np.arctan2(finalx[:,0],finalx[:,1])
    z = finalx[:, 1]**2 + finalx[:, 0]**2
    fig = go.Figure(data=[go.Scatter3d(x=u, y=v, z=z,
                                       mode='markers', marker=dict(
                                           size=2,
                                           color=z,                # set color to an array/list of desired values
                                           colorscale='Viridis',   # choose a colorscale
                                           opacity=0.8
                                       ))])
    fig.update_layout(scene=dict(
        xaxis_title='theta',
        yaxis_title='thetadot',
        zaxis_title='final deviation'),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10))
    fig.show()


def plotly_3d_surf(V, sys_name, slice_idx=None):

    # n_radii = 8
    n_angles = 100

    radii = np.linspace(0.01, 2.0, n_angles)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius.
    # angles = np.repeat(angles[..., np.newaxis], n_angles, axis=1)

    x = list(V.GetVariables())
    CS = np.vstack((np.cos(angles), np.sin(angles)))


# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.

# Compute z to make the pringle surface.
    z = samples.evaluate(radii, CS, V, x, slice_idx)

    x=(radii*np.cos(angles)).flatten()
    y=(radii*np.sin(angles)).flatten()



# u = np.linspace(0, 2*np.pi, 24)
# v = np.linspace(-1, 1, 8)
# u,v = np.meshgrid(u,v)
# u = u.flatten()
# v = v.flatten()

# tp = 1 + 0.5*v*np.cos(u/2.)
# x = tp*np.cos(u)
# y = tp*np.sin(u)
# z = 0.5*v*np.sin(u/2.)

# points2D = np.vstack([u,v]).T
# tri = Delaunay(points2D)
# simplices = tri.simplices

    fig1 = FF.create_trisurf(x=x, y=y, z=z,
                         colormap="Portland",
                         simplices=simplices,
                         title="Mobius Band")
    py.iplot(fig1, filename="Mobius-Band")
