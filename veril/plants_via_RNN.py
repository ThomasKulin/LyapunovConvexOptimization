import numpy as np
import six
from keras import backend as K
# from control.matlab import drss
# from scipy import integrate
# import matplotlib.pyplot as plt
# import os
# import tensorflow as tf

# TODO: differentiate between true CT and foward-euler CT (true DT converted)


def get(plant_name, dt, obs_idx):
    if isinstance(plant_name, six.string_types):
        identifier = str(plant_name)
        return globals()[identifier](dt=dt, obs_idx=obs_idx)


class Plant:

    def __init__(self):
        pass

    def obs(self, obs_idx):
        self.obs_idx = obs_idx
        if obs_idx is None:
            self.num_outputs = self.num_states
        else:
            self.num_outputs = len(self.obs_idx)

    def step(self, x, u):
        pass

    def get_states(self):
        return self.states

    def get_obs(self, x):
        if self.obs_idx is None:
            return x
        else:
            # TODO: buggy here (dimensionaliy mismatch)
            Warning('WIP')
            if K.is_tensor(x):
                return K.dot(K.constant(np.eye(self.num_states)
                                        [self.obs_idx]), x)
            else:
                return x[self.obs_idx]

    def ydot(self, x, u):
        xdot = self.xdot(x, u)
        if self.obs_idx is None:
            return xdot
        else:
            return xdot[self.obs_idx]

    def get_data(self, num_samples, timesteps, num_units, lb=-1, ub=1, random_c=False):
        init_x = np.random.uniform(lb, ub, (num_samples, self.num_states))
        if random_c:
            init_c = np.random.uniform(lb, ub, (num_samples, num_units))
        else:
            init_c = np.zeros((num_samples, num_units))
        ext_in = np.zeros((num_samples, timesteps, self.num_disturb))
        x = [init_x, init_c, ext_in]
        y = np.tile(self.y0, (num_samples, 1))
        return x, y


class Pendulum(Plant):

    def __init__(self, dt=1e-3, obs_idx=None, num_disturb=0):
        self.name = 'Pendulum'
        self.m = 1
        self.l = .5
        self.b = 0.1
        self.lc = .5
        self.I = .25
        self.g = 9.81

        # b/c we deal with the [sin cos thetadot] coordinate
        self.num_states = 3
        self.num_inputs = 1
        self.num_disturb = num_disturb

        self.obs(obs_idx)

        self.dt = dt
        self.x0 = np.array([0, -1, 0])
        self.y0 = self.get_obs(self.x0)
        self.u0 = 0
        self.manifold = True

    def step(self, x, u):
        delta_func = lambda s, c, thetadot: [c * thetadot,
                                             -s * thetadot,
                                             (-self.b * thetadot + u) /
                                             (self.m * self.l * self.l) - self.g * s /
                                             self.l]
        if K.is_tensor(x):
            s = K.dot(x, K.constant([1, 0, 0], shape=(3, 1)))
            c = K.dot(x, K.constant([0, 1, 0], shape=(3, 1)))
            thetadot = K.dot(x, K.constant([0, 0, 1], shape=(3, 1)))
            delta = K.concatenate(delta_func(s, c, thetadot))

        else:
            [s, c, thetadot] = x
            delta = np.array(delta_func(s, c, thetadot))

        # desired fixed point should be sin(pi)=0, cos(pi)=-1, thetadot =0
        self.states = x + delta * self.dt
        return self.states

    def A0(self):
        return [[0, thetadot, c],
                [-thetadot, 0, -s],
                [-self.g / self.l, 0, (-self.b) / (self.m * self.l * self.l)]]

    def get_data(self, num_samples, timesteps, num_units, lb=-1, ub=1, random_c=False):
        # u = np.linspace(-np.pi, np.pi, np.sqrt(num_samples))
        # v = np.linspace(-1, 1, np.sqrt(num_samples))
        # u, v = np.meshgrid(u, v)
        # theta, thetadot = u.ravel(), v.ravel()

        theta = np.random.uniform(np.pi - .1, np.pi + .1, (num_samples,))
        thetadot = np.random.uniform(lb, ub, (num_samples,))

        init_x_train = np.array([np.sin(theta), np.cos(theta), thetadot]).T
        if random_c:
            init_c_train = np.random.uniform(lb, ub, (num_samples, num_units))
        else:
            init_c_train = np.zeros((num_samples, num_units))
        ext_in_train = np.zeros((num_samples, timesteps, self.num_disturb))
        x_train = [init_x_train, init_c_train, ext_in_train]
        y_train = np.tile(self.y0, (num_samples, 1))
        return x_train, y_train

    def get_manifold(self, x):
        return x[0]**2 + x[1]**2 - 1


class Satellite(Plant):

    def __init__(self, dt=1e-3, obs_idx=None, num_disturb=0):
        self.name = 'Satellite'
        self.num_states = 6
        self.num_inputs = 3
        self.obs(obs_idx)
        self.dt = dt
        self.num_disturb = num_disturb
        self.x0 = np.zeros((self.num_states,))
        self.y0 = self.get_obs(self.x0)
        self.u0 = 0
        self.manifold = False
    # def reset(self, lb=-2.5, ub=2.5):
        # np.random.seed(3)
        # self.states = np.random.uniform(lb, ub, (1,self.num_states))
        # self.states = K.random_uniform((self.num_states,), lb, ub)

    def sigma(self, alpha):
        # accepts alpha of shape (None,3)
        # [a1, a2, a3] = alpha
        # return np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

        sigma = K.variable([[0, -alpha[0, 2], alpha[0, 1]],
                            [alpha[0, 2], 0, -alpha[0, 0]],
                            [-alpha[0, 1], alpha[0, 0], 0]])
        return sigma

    def cross(self, u, v):
        u1 = K.dot(u, K.constant([1, 0, 0], shape=(3, 1)))
        u2 = K.dot(u, K.constant([0, 1, 0], shape=(3, 1)))
        u3 = K.dot(u, K.constant([0, 0, 1], shape=(3, 1)))
        v1 = K.dot(v, K.constant([1, 0, 0], shape=(3, 1)))
        v2 = K.dot(v, K.constant([0, 1, 0], shape=(3, 1)))
        v3 = K.dot(v, K.constant([0, 0, 1], shape=(3, 1)))
        # u1, u2, u3 = tf.split(u, 3)
        # v1, v2, v3 = tf.split(v, 3)
        return K.concatenate([(u2 * v3) - (u3 * v2), (u3 * v1) - (u1 * v3),
                              (u1 * v2) - (u2 * v1)])

    def step(self, x, u):
        H = K.constant(np.diag([2, 1, .5]))
        H_inv = K.constant(np.diag([.5, 1, 2]))
        eye3 = K.constant(np.eye(3))

        # w = x[:, :3]
        # phi = x[:, 3:6]
        w = K.dot(x, K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],
                                 [0, 0, 0], [0, 0, 0]], shape=(6, 3)))
        phi = K.dot(x, K.constant([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                                   [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  shape=(6, 3)))
        # CT dynamics:
        # H*dot(w)=-sigma(w)*H*w+u, or equivalently:
        # dot(w)=H_inv*(-sigma(w)*H*w+u), forward Euler
        # w2=w+(H_inv.dot(np.dot(-self.sigma(w),np.dot(H,w))+u))*self.dt
        _ = K.transpose(K.dot(H, K.transpose(w)))
        _ = self.dt * K.dot(H_inv, -K.transpose(self.cross(w, _)))
        _ = K.transpose(_)
        w2 = (w + _) + K.transpose(K.dot(H_inv, K.transpose(u)) * self.dt)

        # dot(phi)=.5*(eye+phi*phi.T+sigma(phi))*w
        _ = K.dot(K.transpose(phi), phi)
        _ = .5 * K.dot(_ + eye3, K.transpose(w)) * self.dt
        _ = _ + .5 * K.transpose(self.cross(phi, w)) * self.dt
        phi2 = phi + K.transpose(_)

        # for comparing dot and cross:
        # self._compare_dotcross(x,u)
        # print((K.eval(K.concatenate([w2,phi2]))))

        # for comparing keras and numpy behaviors:
        # np_u = np.array([4,8,7])
        # np_u = np.reshape(np_u,(3,1))
        # np_w2=w+(H_inv.dot(np.dot(-self.sigma(w),np.dot(H,w))+np_u))*self.dt
        # tensor_w2=K.eval(w2)

        self.states = K.concatenate([w2, phi2])
        return self.states

    def _get_openu(self):
        return self.open_u

    def _compare_dotcross(self, x, u):
        H = K.constant(np.diag([2, 1, .5]))
        H_inv = K.constant(np.diag([.5, 1, 2]))
        eye3 = K.constant(np.eye(3))
        w = x[:, :3]
        phi = x[:, 3:6]
        # CT dynamics:
        # H*dot(w)=-sigma(w)*H*w+u, or equivalently:
        # dot(w)=H_inv*(-sigma(w)*H*w+u), forward Euler
        # w2=w+(H_inv.dot(np.dot(-self.sigma(w),np.dot(H,w))+u))*self.dt
        # _ = np.linalg.multi_dot([H_inv,-self.sigma(w),H,w])*self.dt
        # dot prod realization
        _ = K.transpose(K.dot(K.dot(K.dot(H_inv, -self.sigma(w)), H),
                              K.transpose(w))) * self.dt
        w2 = (w + _) + K.transpose(K.dot(H_inv, K.transpose(u)) * self.dt)

        # dot(phi)=.5*(eye+phi*phi.T+sigma(phi))*w
        # dot product realization
        phi2 = phi + K.transpose(.5 * K.dot((eye3 + self.sigma(phi) +
                                             K.dot(K.transpose(phi), phi)),
                                            K.transpose(w))) * self.dt
        print(K.eval(K.concatenate([w2, phi2])))


class DoubleIntegrator(Plant):

    def __init__(self, dt=1, obs_idx=None, num_disturb=0):
        self.name = 'DoubleIntegrator'
        self.num_states = 2
        self.num_inputs = 1
        self.obs(obs_idx)
        self.num_disturb = num_disturb
        self.x0 = np.zeros((self.num_states,))
        self.y0 = self.get_obs(self.x0)
        self.u0 = 0
        self.manifold = False
        self.dt = dt

    def step(self, x, u):
        x1 = K.dot(x, K.constant([1, 0], shape=(2, 1)))
        x2 = K.dot(x, K.constant([0, 1], shape=(2, 1)))

        x1_next = x1 + x2
        x2_next = x2 + u
        self.states = K.concatenate([x1_next, x2_next])
        return self.states

    def xdot(self, x, u):
        A = np.array([[1, 1], [0, 1]])
        B = np.array([[0], [1]])
        xplus = A@x + B@u
        return (xplus - x) / self.dt

    def sim_traj(self, timesteps, full_states, stable_sample, u=0,
                 scale_time=1, given_initial=None):
        if given_initial is None:
            self.reset()
        else:
            self.states = given_initial
        return np.array([self.step_once(full_states, u) for i in range
                         (timesteps)])

    def reset(self):
        self.states = np.random.uniform(-5, 5, (self.num_states,))

    def plot_sim_traj(self, timesteps, full_states, stable_sample,
                      num_samples=1, scale_time=1):
        fig, ax = plt.subplots()
        for i in range(num_samples):
            sim_traj = self.sim_traj(timesteps, full_states,
                                     stable_sample, scale_time=scale_time)
            if full_states:
                ax.scatter(sim_traj[:, 0], sim_traj[:, 1], c='b')
                ax.scatter(sim_traj[0, 0], sim_traj[0, 1], c='g')
                # ax.axis([-3, 3, -3, 3])
                plt.xlabel('$x_1$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
            else:
                t = np.arange(timesteps)
                ax.plot(t, sim_traj, c='b')
                plt.xlabel('$t$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
        plt.show()


class DubinsPoly(Plant):

    def __init__(self, dt=1e-4, obs_idx=None, num_disturb=0):
        self.name = 'DubinsPoly'
        self.num_states = 3
        self.num_inputs = 2
        self.obs(obs_idx)
        self.num_disturb = num_disturb
        self.x0 = np.zeros((self.num_states,))
        self.y0 = self.get_obs(self.x0)
        self.manifold = False
        self.dt = dt

    def step(self, x, u):
        # coordinate changed to polynomial, see:
        # Kinematic and Dynamic Control of a Wheeled Mobile Robot
        # David DeVon and Timothy Bretl
        x1 = K.dot(x, K.constant([1, 0, 0], shape=(3, 1)))
        x2 = K.dot(x, K.constant([0, 1, 0], shape=(3, 1)))
        x3 = K.dot(x, K.constant([0, 0, 1], shape=(3, 1)))
        u1 = K.dot(u, K.constant([1, 0], shape=(2, 1)))
        u2 = K.dot(u, K.constant([0, 1], shape=(2, 1)))
        x1_next = u1
        x2_next = u2
        x3_next = x2 * u1
        delta = K.concatenate([x1_next, x2_next, x3_next])
        return x + delta * self.dt

    def xdot(self, x, u):
        return np.array([u[0], u[1], x[1] * u[0]])


class DubinsTrig(Plant):

    def __init__(self, dt=1e-4, obs_idx=None, num_disturb=0):
        self.name = 'DubinsTrig'
        self.num_states = 3
        self.num_inputs = 2
        self.obs(obs_idx)
        self.num_disturb = num_disturb
        self.x0 = np.zeros((self.num_states,))
        self.y0 = self.get_obs(self.x0)
        self.manifold = False
        self.dt = dt

    def step(self, x, u):
        x1 = K.dot(x, K.constant([1, 0, 0], shape=(3, 1)))
        x2 = K.dot(x, K.constant([0, 1, 0], shape=(3, 1)))
        x3 = K.dot(x, K.constant([0, 0, 1], shape=(3, 1)))
        u1 = K.dot(u, K.constant([1, 0], shape=(2, 1)))
        u2 = K.dot(u, K.constant([0, 1], shape=(2, 1)))
        x1_next = u1 * K.cos(x3)
        x2_next = u1 * K.sin(x3)
        x3_next = u2
        delta = K.concatenate([x1_next, x2_next, x3_next])
        return x + delta * self.dt

    def xdot(self, x, u):
        return np.array([u[0] * x[2], u[0] * x[2], u[1]])


class HybridPlant():

    def __init__(self, num_states=2, num_inputs=1, num_outputs=1):
        self.name = 'HybridPlant'
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def step_once(self, full_states):
        x1, x2 = self.states
        if x2 <= 0:
            x1_next = .5 * x1 + x1 * x2
            x2_next = -0.8 * x2
        else:
            x1_next = 0.5 * x1
            x2_next = -0.8 * x2 - x1**2
        self.states = np.array([x1_next, x2_next])
        return self._get_obs(full_states)

    def sim_traj(self, timesteps, full_states, stable_sample,
                 scale_time=1, given_initial=None):
        if given_initial is None:
            self.reset()
        else:
            self.states = given_initial
        return np.array([self.step_once(full_states) for i in range(timesteps)])

    def _get_obs(self, full_states):
        if full_states:
            return self.states
        else:
            # return last states
            return [self.states[-1]]

    def reset(self):
        self.states = np.random.uniform(-1, 1, (self.num_states,))

    def plot_sim_traj(self, timesteps, full_states, stable_sample,
                      num_samples=1, scale_time=1):
        fig, ax = plt.subplots()
        for i in range(num_samples):
            sim_traj = self.sim_traj(timesteps, full_states,
                                     stable_sample, scale_time=scale_time)
            if full_states:
                ax.scatter(sim_traj[:, 0], sim_traj[:, 1], c='b')
                ax.scatter(sim_traj[0, 0], sim_traj[0, 1], c='g')
                # ax.axis([-3, 3, -3, 3])
                plt.xlabel('$x_1$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
            else:
                t = np.arange(timesteps)
                ax.plot(t, sim_traj, c='b')
                plt.xlabel('$t$')
                plt.ylabel('$x_2$', rotation=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
        plt.show()

    def plot_sampled_ROA(self, full_states):
        fig, ax = plt.subplots()
        num = 100
        u = np.linspace(-3, 3, num=num)
        v = np.linspace(-3, 3, num=num)
        u, v = np.meshgrid(u, v)
        u, v = u.ravel(), v.ravel()
        x = np.array([u, v]).T

        for i in range(x.shape[0]):
            traj = []
            self.states = x[i]
            for j in range(20000):
                if (np.abs(self.states) <= 1e3).all():
                    next_obs = self.step_once(full_states)
                    traj.append(next_obs)
                else:
                    x[i] = np.NaN
                    traj = None
                    break
            if traj is not None:
                traj = np.array(traj)
                if (np.abs(traj[:, -1]) >= 1e-1).any():
                    x[i] = np.NaN

        ax.scatter(x[:, 0], x[:, 1])
        plt.show()


class LinearPlant():

    def __init__(self, num_states, num_inputs=1, num_outputs=1):
        self.name = 'Linear_dim' + str(num_states)
        self.num_states = num_states
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        sys = drss(states=num_states)
        self.A = np.array(sys.A)
        self.initial_states = None
        self.open_u = np.zeros((num_inputs,))

    def step_once(self, full_states):
        self.states = np.dot(self.A, self.states)
        return self._get_obs(full_states)

    def sim_traj(self, timesteps, full_states, stable_sample):
        self.reset()
        return np.array([self.step_once(full_states) for i in range(timesteps)])

    def _get_obs(self, full_states):
        if full_states:
            return self.states
        else:
            # return last states
            return [self.states[-1]]

    def step(self):
        self.states = np.dot(self.A, self.states)
        return self._get_obs()

    def reset(self):
        self.initial_states = np.random.randn(self.num_states,)
        self.states = self.initial_states
