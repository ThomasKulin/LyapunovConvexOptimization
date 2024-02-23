import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from . import samples
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.pyplot import cm
import matplotlib.lines as mlines


def plot_params(system, **kwargs):
    sys_name = system.name
    add_title = kwargs['add_title'] if 'add_title' in kwargs else ''
    slice_idx = kwargs['slice_idx'] if 'slice_idx' in kwargs else system.slice

    id1, id2 = str(slice_idx[0] + 1), str(slice_idx[1] + 1)
    file_dir = '../' + sys_name
    stableSamples_path = file_dir + '/stableSamplesSlice' + id1 + id2 + '.npy'
    if os.path.exists(stableSamples_path):
        stable_samples = np.load(stableSamples_path, allow_pickle=True)
    else:
        stable_samples = None

    return [sys_name, slice_idx, file_dir, add_title, id1, id2, stable_samples]


def plot_funnel(V, system, **kwargs):
    [sys_name, slice_idx, file_dir, add_title, id1, id2, stable_samples] = \
        plot_params(system, **kwargs)

    if sys_name is 'VanderPol':
        xlim = np.load(file_dir + '/VanDerPol_limitCycle.npy')
        plt.plot(xlim[0], xlim[1], color='red', label='Known ROA Boundary')
    elif stable_samples is not None:
        plt.scatter(stable_samples[:, 0], stable_samples[:, 1], color='red',
                    label='Simulated Stable Samples')

    x = samples.levelsetData(V, slice_idx)[0]
    plt.plot(x[:, 0], x[:, 1], label='Verified ROA Boundary')

    plt.xlabel('X' + id1)
    plt.ylabel('X' + id2)
    leg = plt.legend()
    plt.title(sys_name + add_title)
    plt.savefig(file_dir + '/funnel' + add_title + '.png', dpi=None,
                facecolor='w', edgecolor='w', orientation='portrait',
                format=None, transparent=False,
                bbox_inches=None, pad_inches=0.1)
    plt.show()

def plot_samples(system, **kwargs):
    [sys_name, slice_idx, file_dir, add_title, id1, id2, stable_samples] = \
        plot_params(system, **kwargs)

    fig, ax = plt.subplots()  # Create a figure and an axis object

    if stable_samples is not None:
        ax.scatter(stable_samples[:, 0], stable_samples[:, 1], color='red',
                   label='Simulated Stable Samples')

    if 'xaxis' in kwargs:
        ax.set_xlim(kwargs['xaxis'])  # Set x-axis limits
    if 'yaxis' in kwargs:
        ax.set_ylim(kwargs['yaxis'])  # Set y-axis limits

    ax.set_xlabel('X' + id1)
    ax.set_ylabel('X' + id2)
    leg = ax.legend()
    ax.set_title(sys_name + add_title)

    plt.savefig(file_dir + '/sampledROA' + add_title + '.png', dpi=None,
                facecolor='w', edgecolor='w', orientation='portrait',
                format=None, transparent=False,
                bbox_inches=None, pad_inches=0.1)
    plt.show()



def scatterSamples(samples, system, **kwargs):
    [sys_name, slice_idx, file_dir, add_title, id1, id2, stable_samples] = \
        plot_params(system, **kwargs)

    if stable_samples is not None:
        plt.scatter(stable_samples[:, 0], stable_samples[:, 1], color='red',
                    label='Simulated Stable Samples')

    if sys_name is 'VanderPol':
        xlim = np.load(file_dir + '/VanderPol_limitCycle.npy')
        bdry = plt.plot(xlim[0], xlim[1], color='yellow', label='ROA boundary')

    plt.scatter(samples[:, slice_idx[0]], samples[:, slice_idx[1]],
                label='Samples')

    xlab, ylab = 'X' + id1, 'X' + id2
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.title(sys_name + add_title)
    plt.show()





def plot_traj(initial, system, **kwargs):
    [sys_name, slice_idx, file_dir, add_title, id1, id2, stable_samples] = \
        plot_params(system, **kwargs)
    # sys_name = system.name
    nx = system.num_states
    # add_title = kwargs['add_title'] if 'add_title' in kwargs else ''

    n = initial.shape[0]
    color = cm.coolwarm(np.linspace(0, 1, n))
    # markers = ['o', 'P', 'X', 0, 'v', 'h', '^', '.', ',', '<', '>', '1', '2',
    #            '3', '4', '8', 's', 'p', '*',  'H', '+', 'x', 'D', 'd', '|',
    #            '_', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'None', None, ' ', '']
    # markers = markers[:nx]
    # handles = [mlines.Line2D([], [], marker=markers[i], linestyle='None',
    #                          markersize=8, label='x' + str(i + 1)) for i in
    #            range(nx)]
    # plt.legend(handles=handles)
    slice_idx = kwargs['slice_idx'] if 'slice_idx' in kwargs else range(nx)
    fig, axs = plt.subplots(1, len(slice_idx), sharey=False, figsize=(9, 3))
    if 'V' in kwargs: fig2, axs2 = plt.subplots()
    st = fig.suptitle("\n" + sys_name + ' Simulation ' + add_title)
    [j.set_ylabel('x' + str(i + 1)) for (i, j) in zip(slice_idx, axs)]
    [i.set_xlabel('time') for i in axs]
    final_states = []
    final_Vs = []
    for i, c in zip(initial, color):
        sol = system.forward_sim(i, **kwargs)
        if sol.status != -1:
            [j.plot(sol.t, sol.y[i],  c=c) for (i, j) in zip(slice_idx, axs)]
            final_states.append(sol.y[:, -1])
            if 'V' in kwargs:
                Vtraj = system.get_v_values(sol.y.T, V=kwargs['V'])
                axs2.plot(sol.t, Vtraj, c=c)
                axs2.set_xlabel('time')
                axs2.set_ylabel('V')
                axs2.set_title('V trajectory')
                print('final V value is %s' % Vtraj[-1])
                final_Vs.append(Vtraj[-1])
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.8)
    fig.savefig(file_dir + '/Sim.png')
    plt.show()
    return final_states, final_Vs


def plot3d(V, system, r_max=[3, 3], slice_idx=(0, 1), in_xo=True, limit=None,
           nd=151):
    [sys_name, slice_idx, file_dir, add_title, id1, id2, stable_samples] = \
        plot_params(system, slice_idx=slice_idx)
    if in_xo and hasattr(system, 'poly_to_orig'):
        V, _ = system.poly_to_orig(V)
        sym_x = system.xo
    else:
        sym_x = system.sym_x

    nq = nd
    nqd = nd
    x = np.linspace(-r_max[0], r_max[0], nq)
    y = np.linspace(-r_max[1], r_max[1], nqd)
    X, Y = np.meshgrid(x, y)
    Z = X.copy()
    for i in range(nq):
        for j in range(nqd):
            env = dict(zip(sym_x, np.zeros(system.num_states,)))
            env1 = {sym_x[slice_idx[0]]: X[i, j], sym_x[slice_idx[1]]: Y[i, j]}
            env.update(env1)
            Z[i, j] = V.Evaluate(env)
    # if limit is not None:
    #     X3d = X[np.abs(Z) <= limit]
    #     Y3d = Y[np.abs(Z) <= limit]
    #     Z3d = Z[np.abs(Z) <= limit]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0.2, cmap=cm.coolwarm,
                    antialiased=True)
    # ax.contour(X, Y, Z, zdir='z', offset=min(Z.ravel())-.25, cmap=cm.coolwarm)
    # if limit is not None: ax.set_zlim(min(Z.ravel())-.25, limit)

    fig2, ax2 = plt.subplots()
    ax2.contour(X, Y, Z, levels=30, cmap=cm.Spectral)
    xlab = 'X' + str(slice_idx[0] + 1)
    ylab = 'X' + str(slice_idx[1] + 1)
    [i.set_xlabel(xlab) for i in [ax, ax2]]
    [i.set_ylabel(ylab) for i in [ax, ax2]]
    fig.savefig(file_dir + '/3d' + id1 + id2 + '.png', dpi=None, facecolor='w',
                edgecolor='w', orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    fig2.savefig(file_dir + '/contour' + id1 + id2 + '.png', dpi=None,
                 facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                 format=None, transparent=False, bbox_inches='tight', pad_inches=0.1)

    plt.show()

# def phase_portrait(self, ax, ax_max):
#     num = 60
#     u = np.linspace(-ax_max, ax_max, num=num)
#     v = np.linspace(-ax_max, ax_max, num=num)
#     u, v = np.meshgrid(u, v)
#     u, v = u.ravel(), v.ravel()
#     x = -v
#     y = u - v + v * u**2
#     # angles='xy', width=1e-3,scale_units='xy', scale=12, color='r'
#     ax.quiver(u, v, x, y, color='r', width=1e-3, scale_units='x')

# def plot_sim_traj(self, timesteps, full_states, stable_sample,
#                   num_samples=1, scale_time=1):
#     fig, ax = plt.subplots()
#     for i in range(num_samples):
#         sim_traj = self.sim_traj(timesteps, full_states,
#                                  stable_sample, scale_time=scale_time)
#         if full_states:
#             self._phase_portrait(ax, 3)
#             ax.scatter(sim_traj[:, 0], sim_traj[:, 1], c='b')
#             ax.scatter(sim_traj[0, 0], sim_traj[0, 1], c='g')
#             # ax.axis([-3, 3, -3, 3])
#             plt.xlabel('$x_1$')
#             plt.ylabel('$x_2$', rotation=0)
#             plt.xticks(fontsize=8)
#             plt.yticks(fontsize=8)
#         else:
#             t = np.arange(timesteps)
#             ax.scatter(t, sim_traj, c='b')
#             plt.xlabel('$t$')
#             plt.ylabel('$x_2$', rotation=0)
#             plt.xticks(fontsize=8)
#             plt.yticks(fontsize=8)
#     plt.show()
