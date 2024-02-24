# import sys
# sys.path.append(
#     "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
from pydrake.all import (MathematicalProgram, Polynomial, Expression,
                         MonomialBasis, Jacobian, Evaluate, sin, cos)

import os
import itertools
import six
# import time
import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov
from scipy import integrate


class ClosedLoopSys(object):

    def __init__(self, name, num_states, idx=(0, 1)):
        self.loop_closed = True

        self.name = name
        self.num_states = num_states
        self.slice = idx
        self.x0 = np.zeros((num_states,))
        self.x0dot = np.zeros((num_states,))
        self.all_slices = list(itertools.combinations(range(num_states), 2))
        if hasattr(self, 'special_fixed_pt'):
            self.special_fixed_pt()

        self.at_fixed_pt_tol = 1e-2
        self.int_stop_ub = 15
        self.int_horizon = 10
        self.d = 2
        self.num_grid = 100

    def init_x_f(self):
        prog = MathematicalProgram()
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")
        self.sym_f = self.fx(None, self.sym_x)

    def set_syms(self, deg_ftrs, deg_u, rm_one):
        self.rm_one = rm_one
        self.deg_ftrs = deg_ftrs
        self.degV = 2 * deg_ftrs
        x = self.sym_x
        xbar = self.sym_x - self.x0
        self.sym_phi = get_monomials(xbar, deg_ftrs, rm_one)
        sym_dphidx = Jacobian(self.sym_phi, x)

        if self.loop_closed:
            self.sym_eta = sym_dphidx@self.sym_f
        else:
            self.deg_u = deg_u
            self.sym_dphidx = sym_dphidx
            self.sym_ubasis = get_monomials(xbar, deg_u, True)

    def features_at_x(self, x, file_path):  # x: (num_samples, sys_dim)
        phi, eta = [], []
        for i in x:
            env = dict(zip(self.sym_x, i))
            phi.append([j.Evaluate(env) for j in self.sym_phi])
            eta.append([j.Evaluate(env) for j in self.sym_eta])
        features = [np.array(phi), np.array(eta)]
        if file_path is not None:
            np.savez_compressed(file_path, phi=features[0], eta=features[1])
        return features

    def set_sample_variety_features(self, V):
        # this requires far lower degreed multiplier xxd and consequentially
        # lower degree psi, re-write both
        # rm_one = self.rm_one
        x = self.sym_x
        xbar = self.sym_x - self.x0
        self.sym_V = V
        self.sym_Vdot = self.set_Vdot(V)
        self.degVdot = self.degV - 1 + self.degf
        self.deg_xxd = int(np.ceil((self.degVdot - self.degV) / 2))
        self.sym_xxd = (xbar.T@xbar)**(self.deg_xxd)
        self.deg_psi = int(((2 * self.deg_xxd + self.degV) / 2))
        self.sym_psi = get_monomials(xbar, self.deg_psi, True)

    def get_sample_variety_features(self, samples):
        # samples: (num_samples, sys_dim)
        xbar = samples - self.x0
        xxd = [(i.T@i)**(self.deg_xxd) for i in xbar]
        psi = [get_monomials(i, self.deg_psi, True) for i in xbar]
        return [np.array(xxd), np.array(psi)]

    def get_v_values(self, samples, V=None):
        if V is None:
            V = self.sym_V
        return np.array([V.Evaluate(dict(zip(self.sym_x, i))) for i in samples])

    def linearized_quadractic_V(self):
        x = self.sym_x
        try:
            f = self.sym_f
        except:
            f = self.sym_g
        env = dict(zip(x, self.x0))
        A = np.array([[i.Evaluate(env) for i in j]for j in Jacobian(f, x)])
        print('A  %s' % A)
        print('eig of the linearized A matrix %s' % (eig(A)[0]))
        P = solve_lyapunov(A.T, -np.eye(x.shape[0]))
        print('eig of P %s' % (eig(P)[0]))
        # assert (np.all(eig(P)[0] >= 0))
        V = (self.sym_x - self.x0).T@P@(self.sym_x - self.x0)
        return A, P, V

    def P_to_V(self, P, samples=None):
        V0 = self.sym_phi.T@P@self.sym_phi
        Vdot0 = self.set_Vdot(V0)
        self.VdotHessian(Vdot0)
        if samples is not None:
            V_evals = self.get_v_values(samples, V=V0)
            m = np.percentile(V_evals, 75)
            V = V0 / m
        else:
            V = V0
        Vdot = self.set_Vdot(V)
        return V, Vdot

    def VdotHessian(self, Vdot):
        H = Jacobian(Vdot.Jacobian(self.sym_x).T, self.sym_x)
        env = dict(zip(self.sym_x, self.x0))
        H = np.array([[i.Evaluate(env) for i in j]for j in H])
        hessian_eigs = eig(H)[0]
        # print('eig of Hessian of Vdot0 %s' % hessian_eigs)
        return hessian_eigs
        # assert (np.all(eig(H)[0] <= 0))

    def set_Vdot(self, V):
        return V.Jacobian(self.sym_x)@self.sym_f

    def forward_sim(self, x, **kwargs):
        event = kwargs['event'] if 'event' in kwargs else self.event
        int_horizon = kwargs['int_horizon'] if 'int_horizon' in kwargs else \
            self.int_horizon
        sol = integrate.solve_ivp(self.fx, [0, int_horizon], x, events=event)
        return sol

    def random_sample(self, n):
        return np.random.randn(n, self.num_states)

    def sample_stable_inits(self, x=None, **kwargs):
        event = self.event
        if x is None:
            x = self.get_x(**kwargs)
        stableSamples = []
        for i in x:
            # start = time.time()
            sol = self.forward_sim(i, **kwargs)
            # if sol.status == 1:
            # print('event stopping')
            # if sol.status != 1:
            #     print('init %s' % i)
            #     print('final %s' % sol.y[:, -1])
            test = self.is_at_fixed_pt(sol.y[:, -1])
            test2 = sol.status != -1
            test3 = sol.y[:, -1]
            if sol.status != -1 and self.is_at_fixed_pt(sol.y[:, -1]):
                stableSamples.append(i)
            print(np.where(np.all(x==i,axis=1))[0][0])
            # end = time.time()
            # print(end-start)
        stableSamples = np.array(stableSamples)
        os.makedirs('../'+self.name, exist_ok=True)
        name = '../'+self.name+'/stableSamples'
        if 'slice_idx' in kwargs:
            stableSamples = stableSamples[:, kwargs['slice_idx']]
            name = name + 'Slice' + str(kwargs['slice_idx'][0] + 1) + str(kwargs['slice_idx'][1] + 1) + '.npy'
        np.save(name, stableSamples)
        return stableSamples

    def sample_stable_inits_for_all_slices(self, **kwargs):
        for i in self.all_slices:
            samples = self.sample_stable_inits(slice_idx=i, **kwargs)
            name = '../data/' + self.name + '/stableSamplesSlice' + \
                str(i[0] + 1) + str(i[1] + 1) + '.npy'
            np.save(name, samples)

    def get_x(self, **kwargs):
        d = kwargs['d'] if 'd' in kwargs else self.d
        num_grid = kwargs[
            'num_grid'] if 'num_grid' in kwargs else self.num_grid

        x0 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x0, x0)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([x1, x2]).T  # (num_grid**2,2)
        return x[~np.all(x == self.x0, axis=1)]

    def is_at_fixed_pt(self, x):
        if self.name is not "SphericalIP":
            return np.allclose(x, self.x0, atol=self.at_fixed_pt_tol)
        else:
            return np.allclose(x[[2,3,6,7]], self.x0[[2,3,6,7]], atol=self.at_fixed_pt_tol)
            # return np.allclose(x[[6,7]], self.x0[[6,7]], atol=0.2) and \
            #         np.allclose(x[[2, 3]], self.x0[[2, 3]], atol=0.2)


    def event(self, t, x):
        norm = np.linalg.norm(x)
        # in_ = x - self.x0
        if self.int_stop_ub is not None:
            out_ = norm - self.int_stop_ub
        return out_
    event.terminal = True


class S4CV_Plants(ClosedLoopSys):

    def __init__(self, name, num_states, num_inputs, idx=(0, 1)):
        super().__init__(name, num_states, idx=idx)
        self.num_inputs = num_inputs
        self.loop_closed = False

    def init_x_g_B(self):
        prog = MathematicalProgram()
        self.sym_x = prog.NewIndeterminates(self.num_states, "x")
        self.sym_g = self.gx(self.sym_x)

    def features_at_x(self, x, file_path):  # x: (num_samples, sys_dim)
        if x is None:
            x = self.get_x()
        g = self.gx(x.T).T  # (just so 1st dim is # of samples)
        B, phi, dphidx, ubasis = [], [], [], []
        for i in x:
            env = dict(zip(self.sym_x, i))
            phi.append([j.Evaluate(env) for j in self.sym_phi])
            dphidx.append([[j.Evaluate(env) for j in k]for k in
                           self.sym_dphidx])
            ubasis.append([j.Evaluate(env) for j in self.sym_ubasis])
            if hasattr(self, 'B_noneConstant'):
                B.append(self.hx(i))
        features = [g, np.array(phi), np.array(dphidx), np.array(ubasis)]
        if file_path is not None:
            if hasattr(self, 'B_noneConstant'):
                np.savez_compressed(file_path, g=features[0], B=np.array(B),
                                    phi=features[1], dphidx=features[2],
                                    ubasis=features[3])
                features = [g, np.array(B), np.array(phi), np.array(dphidx),
                            np.array(ubasis)]
            else:
                np.savez_compressed(file_path, g=features[0], phi=features[1],
                                    dphidx=features[2], ubasis=features[3])
        return features

    def fx(self, t, y):
        u_basis = get_monomials(y - self.x0, self.deg_u, True)
        u = (u_basis@self.u_weights).T
        # u = np.zeros((self.num_inputs))
        num_sol = self.gx(y) + self.hx(y)@u
        # print(num_sol-[i.Evaluate(dict(zip(self.sym_x,y))) for i in
        #    self.sym_f])
        # print(num_sol)
        return num_sol

    def close_the_loop(self, u_weights):
        self.loop_closed = True
        self.u_weights = u_weights
        self.u = (self.sym_ubasis@u_weights).T
        self.sym_f = self.sym_g + self.hx(self.sym_x)@self.u
        self.degf = max([Polynomial(i, self.sym_x).TotalDegree() for i in
                         self.sym_f])


class PendulumTrig(S4CV_Plants):

    def __init__(self):
        super().__init__('PendulumTrig', 2, 1)
        # parameters
        self.m = 1
        self.l = .5
        self.b = 0.1
        self.lc = .5
        self.I = .25
        self.g = 9.81
        self.init_x_g_B()  # theta=pi, thetadot=0

    def get_x(self, d=2, num_grid=200, slice_idx=None):
        x1 = np.linspace(-np.pi, np.pi, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([x1, x2]).T  # (num_grid**2,2)
        return x[~np.all(x == self.x0, axis=1)]

    def gx(self, x):
        # m l² θ̈=u-m g l sin θ-b θ̇
        [x1, x2] = x
        # thetaddot = -self.b * x2 / self.I - self.g * np.sin(x1) / self.l
        # put the origin at the top right
        x2dot = -self.b * x2 / self.I - self.g * np.sin(x1 + np.pi) / self.l
        return np.array([1 * x2, x2dot])

    def hx(self, x):
        return np.array([[0], [1 / self.I]])

    def is_at_fixed_pt(self, x):
        vel_close = np.isclose(x[-1], 0, atol=self.at_fixed_pt_tol)
        if not vel_close:
            return False
        else:
            y = np.arctan2(np.sin(x[0]), np.cos(x[0]))
            return np.isclose(y, 0, atol=self.at_fixed_pt_tol)

    # for debugging only, comment out
    # def fx(self, t, y):
        # u = (9.81* np.sin(y[0] + np.pi)).reshape((1,))
        # u=np.zeros((1,))
        # num_sol = self.gx(y) + self.hx(y)@u
        # return num_sol


class PendulumRecast(S4CV_Plants):

    def __init__(self):
        super().__init__('PendulumRecast', 3, 1, idx=(1, 2))
        # parameters
        self.m = 1
        self.l = .5
        self.b = 0.1
        self.lc = .5
        self.I = .25
        self.g = 9.81
        self.init_x_g_B()
        self.all_slices = list(itertools.combinations(range(2), 2))
        self.recast = self.sym_x[0]**2 + self.sym_x[1]**2 - 1

    def special_fixed_pt(self):
        prog = MathematicalProgram()
        self.xo = prog.NewIndeterminates(2, "t")
        self.xo0 = [np.pi, 0]
        self.x0 = np.array([0, -1, 0])

    def get_x(self, d=2, num_grid=200, slice_idx=None):
        x1 = np.linspace(-np.pi, np.pi, num_grid)
        x2 = np.linspace(-d, d, num_grid)
        x1, x2 = np.meshgrid(x1, x2)
        x1, x2 = x1.ravel(), x2.ravel()
        x = np.array([np.sin(x1), np.cos(x1), x2]).T  # (num_grid**2,3)
        return x[~np.all(x == self.x0, axis=1)]

    def gx(self, x):
        # m l² θ̈=u-m g l sin θ-b θ̇
        [s, c, thetadot] = x
        # thetaddot = -self.b * x2 / self.I - self.g * np.sin(x1) / self.l
        # put the origin at the top right
        sdot = c * thetadot
        cdot = -s * thetadot
        thetaddot = -self.b * thetadot / self.I - self.g * s / self.l
        return np.array([sdot, cdot, thetaddot])

    def hx(self, x):
        return np.array([[0], [0], [1 / self.I]])

    # def is_at_fixed_pt(self, x):
    #     vel_close = np.isclose(x[-1], 0, atol=self.at_fixed_pt_tol)
    #     if not vel_close:
    #         return False
    #     else:
    #         y = np.arctan2(x[0], x[1])
    #         return np.isclose(y, np.pi, atol=self.at_fixed_pt_tol)

    def random_sample(self, n):
        x1, x2 = np.random.randn(n,), np.random.randn(n,)
        x = np.array([np.sin(x1), np.cos(x1), x2]).T  # (n,3)
        return x

    def poly_to_orig(self, func=None):
        t = self.xo
        env = dict(zip(self.sym_x, [np.sin(1 * t[0]), np.cos(1 * t[0]), t[1]]))
        if func is not None:
            func = func.Substitute(env)
            return func, env
        else:
            return env

    def VdotHessian(self, Vdot):
        Vdot, _ = self.poly_to_orig(func=Vdot)
        H = Jacobian(Vdot.Jacobian(self.xo).T, self.xo)
        env = dict(zip(self.xo, self.xo0))
        H = np.array([[i.Evaluate(env) for i in j]for j in H])
        hessian_eigs = eig(H)[0]
        # print('eig of Hessian of Vdot0 %s' % hessian_eigs)
        return hessian_eigs
        # assert (np.all(eig(H)[0] <= 0))

    def u_scaling_reg(self):
        base = [1., 1., 100.]
        return np.array([get_monomials(base, self.deg_u, True)])

    def debugging_V(self, V):
        up = [np.array([0, -1, 0])]
        down = np.array([np.array([0, 1, 0])])
        print('up V value %s' % system.get_v_values(up, V))
        print('down V value %s' % system.get_v_values(down, V))



class VanderPol(ClosedLoopSys):

    def __init__(self):
        super().__init__('VanderPol', 2)
        self.init_x_f()
        self.degf = 3
        self.int_stop_ub = 5
        self.int_horizon = 20

    def knownROA(self):
        x = self.sym_x
        x1 = x[0]
        x2 = x[1]
        V = (1.8027e-06) + (0.28557) * x1**2 + (0.0085754) * x1**4 + \
            (0.18442) * x2**2 + (0.016538) * x2**4 + \
            (-0.34562) * x2 * x1 + (0.064721) * x2 * x1**3 + \
            (0.10556) * x2**2 * x1**2 + (-0.060367) * x2**3 * x1
        return V

    def fx(self, t, y):
        return - np.array([y[1], (1 - y[0]**2) * y[1] - y[0]])


class Pendubot(ClosedLoopSys):

    def __init__(self):
        super().__init__('Pendubot', 4, idx=(0, 2))
        self.init_x_f()
        self.degf = 3
        self.int_stop_ub = 15
        self.int_horizon = 20

    def get_x(self, **kwargs):
        d = kwargs['d'] if 'd' in kwargs else self.d
        num_grid = kwargs['num_grid'] if 'num_grid' in kwargs else \
            self.num_grid

        x0 = np.linspace(-d, d, num_grid)
        if 'slice_idx' in kwargs:
            x1, x2 = np.meshgrid(x0, x0)
            x1, x2 = x1.ravel(), x2.ravel()
            x = np.zeros((x1.shape[0], 4))  # (num_grid**4,4)
            x[:, kwargs['slice_idx']] = np.array([x1, x2]).T
        else:
            x1, x2, x3, x4 = np.meshgrid(x0, x0, x0, x0)
            x1, x2, x3, x4 = x1.ravel(), x2.ravel(), x3.ravel(), x4.ravel()
            x = np.array([x1, x2, x3, x4]).T  # (num_grid**4,4)
        return x[~np.all(x == self.x0, axis=1)]

    def fx(self, t, y):
        [x1, x2, x3, x4] = y
        return np.array([1 * x2,
                         782 * x1 + 135 * x2 + 689 * x3 + 90 * x4,
                         1 * x4,
                         279 * x1 * x3**2 - 1425 * x1 - 257 * x2 + 273 * x3**3
                         - 1249 * x3 - 171 * x4])

class CartPole_2D(ClosedLoopSys):
    def __init__(self):
        super().__init__('CartPole_2D', 4, idx=(1, 3))
        self.x0 = [0, np.pi, 0, 0]
        self.init_x_f()
        self.degf = 6
        self.int_stop_ub = 1e5
        self.int_horizon = 15


    def get_x(self, **kwargs):
        d = kwargs['d'] if 'd' in kwargs else self.d
        num_grid = kwargs['num_grid'] if 'num_grid' in kwargs else \
            self.num_grid

        x_ang = np.linspace(-d+np.pi, d+np.pi, num_grid)
        x_lin = np.linspace(-2*d, 2*d, num_grid)
        if 'slice_idx' in kwargs:
            x1, x2 = np.meshgrid(x_ang, x_lin)
            x1, x2 = x1.ravel(), x2.ravel()
            x = np.zeros((x1.shape[0], 4))
            x[:, kwargs['slice_idx']] = np.array([x1, x2]).T
        else:
            # x1, x2, x3, x4 = np.meshgrid(x0, x0, x0, x0)
            x1, x2, x3, x4 = np.meshgrid(x_lin, x_ang, x_lin, x_lin)
            x1, x2, x3, x4 = x1.ravel(), x2.ravel(), x3.ravel(), x4.ravel()
            x = np.array([x1, x2, x3, x4]).T  # (num_grid**4,4)
        return x[~np.all(x == self.x0, axis=1)]

    def fx(self, t, x):
        # 3rd order taylor approximation
        return np.array([1 * x[2],
                         1* x [3],
                         (8.6575761912533048e-14 + 31.230314559697316 * x[0] + 55.825248986050326 * x[2] - 121.27701821396735 * x[3] - 706.94474498941622 * (-3.1415926535897931 + x[1]) + 7.2688694181467198e-15 * (x[0] * (-3.1415926535897931 + x[1])) + 1.6583573985648112e-14 * (x[2] * (-3.1415926535897931 + x[1])) - 2.7717474446398496e-14 * (x[3] * (-3.1415926535897931 + x[1])) + ((-135.41515804552165 * x[2] * pow((-3.1415926535897931 + x[1]), 2)) / 2) + ((-59.354823147438005 * x[0] * pow((-3.1415926535897931 + x[1]), 2)) / 2) + ((-0.10000000000000001 * pow(x[3], 2) * (-3.1415926535897931 + x[1])) / 2) + ((-5.5032490410090018e-13 * pow((-3.1415926535897931 + x[1]), 2)) / 2) + ((1.2246467991473533e-17 * pow(x[3], 2)) / 2) + ((226.33035472510514 * x[3] * pow((-3.1415926535897931 + x[1]), 2)) / 2) + ((4493.7438654480447 * pow((-3.1415926535897931 + x[1]), 3)) / 6)),
                         (1.7074876680513897e-13 + 62.460629119394632 * x[0] + 111.65049797210065 * x[2] - 242.5540364279347 * x[3] - 1394.2694899788323 * (-3.1415926535897931 + x[1]) + 2.2186959788673099e-14 * (x[0] * (-3.1415926535897931 + x[1])) + 4.6840390467770332e-14 * (x[2] * (-3.1415926535897931 + x[1])) - 8.5139251325971065e-14 * (x[3] * (-3.1415926535897931 + x[1])) + ((-382.48081406314407 * x[2] * pow((-3.1415926535897931 + x[1]), 2)) / 2) + ((-181.17027541427066 * x[0] * pow((-3.1415926535897931 + x[1]), 2)) / 2) + ((-0.20000000000000001 * pow(x[3], 2) * (-3.1415926535897931 + x[1])) / 2) + ((-1.6177016226570721e-12 * pow((-3.1415926535897931 + x[1]), 2)) / 2) + ((2.4492935982947065e-17 * pow(x[3], 2)) / 2) + ((695.21474587814487 * x[3] * pow((-3.1415926535897931 + x[1]), 2)) / 2) + ((13209.536200832583 * pow((-3.1415926535897931 + x[1]), 3)) / 6))])

class SphericalIP(ClosedLoopSys):
    def __init__(self):
        super().__init__('SphericalIP', 8, idx=(2, 3))
        self.init_x_f()
        self.degf = 6
        self.int_stop_ub = 40
        self.int_horizon = 15


    def get_x(self, **kwargs):
        if 'xaxis' in kwargs:
            da = kwargs['xaxis'] # Set x-axis limits
        if 'yaxis' in kwargs:
            db = kwargs['yaxis']  # Set y-axis limits
        num_grid = kwargs['num_grid'] if 'num_grid' in kwargs else self.num_grid
        span = kwargs['span'] if 'span' in kwargs else None

        if span:  # grid centered at d with span width
            xa = np.linspace(da[0]-span/2, da[0]+span/2, num_grid)
            xb = np.linspace(db[1]-span/2, db[1]+span/2, num_grid)
        else:
            xa = np.linspace(-da[0], da[0], num_grid)
            xb = np.linspace(-db[1], db[1], num_grid)

        if 'slice_idx' in kwargs:
            x1, x2 = np.meshgrid(xa, xb)
            x1, x2 = x1.ravel(), x2.ravel()
            x = np.zeros((x1.shape[0], self.num_states))
            x[:, kwargs['slice_idx']] = np.array([x1, x2]).T
        else:
            ValueError("Error: please pass a slice index")
        return x[~np.all(x == self.x0, axis=1)]

    def fx(self, t, x):
        # Taylor3_SDSOS_Q2e4 = np.array([x[4],
        #                  x[5],
        #                  x[6],
        #                  x[7],
        #                  (0.025985759659187119 * x[0] + 158.07994174502926 * x[2] - 0.3716336569325065 * x[4] + 6.4587701771726376e-06 * x[5] + 7.5341973451989288 * x[6] + 1.8081490240376541e-05 * x[7] + 8.6101854355325684e-05 * (x[0] * x[1] * x[5]) - 2.2362441675895609e-05 * (x[0] * x[2] * x[4]) + 2.0126197508306046e-05 * (x[0] * x[2] * x[6]) - 1.0294480449040951e-05 * (x[1] * x[2] * x[3]) - 2.219845220968448e-05 * (x[1] * x[2] * x[5]) + 8.8009424710867011e-05 * (x[1] * x[4] * x[5]) - 0.00046710617642045367 * (x[2] * x[3] * x[4]) - 0.0002945314534245098 * (x[2] * x[3] * x[5]) - 0.00013254196247309175 * (x[2] * x[3] * x[6]) + 0.013781653511377088 * (x[2] * x[4] * x[6]) - 0.014349195069241701 * (x[3] * x[4] * x[7]) + 0.00017631162648910149 * (x[3] * x[6] * x[7]) + ((-163.21590736129596 * x[2] * pow(x[3], 2)) / 2) + ((-156.93901379606697 * pow(x[2], 3)) / 6) + ((-8.891341671468707 * pow(x[3], 2) * x[6]) / 2) + ((-1.5140437387672692 * pow(x[3], 2) * x[4]) / 2) + ((-0.46981419348525333 * pow(x[4], 3)) / 6) + ((-0.056986517831026637 * x[2] * pow(x[6], 2)) / 2) + ((-0.028514557695434106 * pow(x[1], 2) * x[2]) / 2) + ((-0.028120879944272436 * pow(x[0], 2) * x[2]) / 2) + ((-0.025090692738814192 * x[0] * pow(x[2], 2)) / 2) + ((-0.016092595799421284 * x[2] * pow(x[5], 2)) / 2) + ((-0.015066799039547983 * x[2] * pow(x[4], 2)) / 2) + ((-0.0073192793988367212 * x[2] * pow(x[7], 2)) / 2) + ((-0.0021314436129659439 * x[4] * pow(x[5], 2)) / 2) + ((-0.00040383728288752351 * pow(x[1], 2) * x[4]) / 2) + ((-0.00014423248254671469 * pow(x[0], 2) * x[4]) / 2) + ((-1.0763431740226093e-05 * pow(x[3], 2) * x[7]) / 2) + ((-6.4587701771726376e-06 * pow(x[2], 2) * x[5]) / 2) + ((-6.4587701771726376e-06 * pow(x[3], 2) * x[5]) / 2) + ((7.3180585001504433e-06 * pow(x[2], 2) * x[7]) / 2) + ((8.6355221724694915e-05 * x[0] * pow(x[5], 2)) / 2) + ((8.7588547593900578e-05 * x[0] * pow(x[7], 2)) / 2) + ((0.0012289624398264913 * x[0] * pow(x[4], 2)) / 2) + ((0.0015211431668420085 * pow(x[2], 2) * x[6]) / 2) + ((0.011308141515977189 * x[0] * pow(x[3], 2)) / 2) + ((0.029388560621793 * pow(x[2], 2) * x[4]) / 2)),
        #                  (0.028881439833938614 * x[1] + 164.31606006519559 * x[3] + 6.4587701771726376e-06 * x[4] - 0.36369515692380849 * x[5] + 1.8081490240376541e-05 * x[6] + 6.4062698640100599 * x[7] - 8.791042233320671e-06 * (x[0] * x[1] * x[2]) + 8.6101854355325684e-05 * (x[0] * x[1] * x[4]) - 0.00011673340560498513 * (x[0] * x[2] * x[3]) + 8.6355221724694915e-05 * (x[0] * x[4] * x[5]) - 9.7320608437667311e-05 * (x[0] * x[4] * x[7]) + 1.9978606988716032e-05 * (x[1] * x[2] * x[6]) - 0.00013747195370884162 * (x[1] * x[5] * x[7]) - 1.5100381917224561 * (x[2] * x[3] * x[4]) - 0.0005104843691054417 * (x[2] * x[3] * x[5]) - 8.8929666394410578 * (x[2] * x[3] * x[6]) - 1.8081490240376538e-05 * (x[2] * x[3] * x[7]) + 2.9785274336034646e-05 * (x[2] * x[4] * x[5]) + 0.014510142966381586 * (x[2] * x[5] * x[6]) - 0.0073192793988367212 * (x[2] * x[6] * x[7]) - 0.015541783691952376 * (x[3] * x[5] * x[7]) + ((-657.96028408747316 * pow(x[3], 3)) / 6) + ((-474.91496247076827 * pow(x[2], 2) * x[3]) / 2) + ((-6.4092722039791701 * pow(x[2], 2) * x[7]) / 2) + ((-6.4044892704425642 * pow(x[3], 2) * x[7]) / 2) + ((-1.7433342425888421 * pow(x[3], 2) * x[5]) / 2) + ((-0.53142479916341423 * pow(x[5], 3)) / 6) + ((-0.024950524189713351 * x[1] * pow(x[2], 2)) / 2) + ((-0.0021314436129659439 * pow(x[4], 2) * x[5]) / 2) + ((-0.00045855276166194248 * pow(x[0], 2) * x[5]) / 2) + ((-0.00030993484769751645 * pow(x[1], 3)) / 6) + ((-9.6422026586060819e-05 * pow(x[1], 2) * x[5]) / 2) + ((-4.9387366140279154e-05 * x[5] * pow(x[7], 2)) / 2) + ((-3.7383833396900723e-05 * pow(x[5], 2) * x[7]) / 2) + ((-1.0763431740226098e-05 * pow(x[3], 2) * x[6]) / 2) + ((-6.4587701771726376e-06 * pow(x[2], 2) * x[4]) / 2) + ((-6.4587701771726376e-06 * pow(x[3], 2) * x[4]) / 2) + ((1.3190444673823373e-05 * pow(x[0], 2) * x[1]) / 2) + ((2.5399548740526985e-05 * pow(x[2], 2) * x[6]) / 2) + ((7.472953457774082e-05 * pow(x[7], 3)) / 6) + ((8.8009424710867011e-05 * x[1] * pow(x[4], 2)) / 2) + ((0.00012372475833795745 * x[1] * pow(x[7], 2)) / 2) + ((0.00017631162648910149 * x[3] * pow(x[6], 2)) / 2) + ((0.0033402252575842608 * x[1] * pow(x[5], 2)) / 2) + ((0.0086970010215202988 * x[3] * pow(x[7], 2)) / 2) + ((0.010830936712579061 * x[1] * pow(x[3], 2)) / 2) + ((0.015943550076935223 * x[3] * pow(x[4], 2)) / 2) + ((0.017268648546613752 * x[3] * pow(x[5], 2)) / 2) + ((0.018765305007514912 * pow(x[1], 2) * x[3]) / 2) + ((0.01938138365082779 * pow(x[0], 2) * x[3]) / 2) + ((0.036790575236646472 * pow(x[2], 2) * x[5]) / 2)),
        #                  ( - 0.028873066287985687 * x[0] - 164.74437971669917 * x[2] + 0.41292628548056276 * x[4] - 7.1764113079695978e-06 * x[5] - 8.3713303835543655 * x[6] - 2.0090544711529489e-05 * x[7] - 9.5668727061472977e-05 * (x[0] * x[1] * x[5]) + 2.4847157417661789e-05 * (x[0] * x[2] * x[4]) - 2.2362441675895609e-05 * (x[0] * x[2] * x[6]) + 0.032101927015986284 * (x[1] * x[2] * x[3]) + 2.4664946899649425e-05 * (x[1] * x[2] * x[5]) - 9.7788249678741122e-05 * (x[1] * x[4] * x[5]) + 0.00052618327399736251 * (x[2] * x[3] * x[4]) - 0.40377847274487111 * (x[2] * x[3] * x[5]) + 0.00016735939190385364 * (x[2] * x[3] * x[6]) + 7.1180776266778443 * (x[2] * x[3] * x[7]) - 0.015312948345974544 * (x[2] * x[4] * x[6]) + 0.015943550076935223 * (x[3] * x[4] * x[7]) - 0.00019590180721011281 * (x[3] * x[6] * x[7]) + ((-1.9918674673346259 * x[2] * pow(x[7], 2)) / 2) + ((-0.44558024172699939 * pow(x[2], 2) * x[4]) / 2) + ((-0.012564601684419098 * x[0] * pow(x[3], 2)) / 2) + ((-0.0013655138220294348 * x[0] * pow(x[4], 2)) / 2) + ((-9.7320608437667311e-05 * x[0] * pow(x[7], 2)) / 2) + ((-9.5950246360772143e-05 * x[0] * pow(x[5], 2)) / 2) + ((7.176411307969597e-06 * pow(x[3], 2) * x[5]) / 2) + ((1.1959368600251213e-05 * pow(x[3], 2) * x[7]) / 2) + ((1.1959368600251218e-05 * pow(x[2], 2) * x[7]) / 2) + ((1.4352822615939194e-05 * pow(x[2], 2) * x[5]) / 2) + ((0.00016025831394079408 * pow(x[0], 2) * x[4]) / 2) + ((0.00044870809209724837 * pow(x[1], 2) * x[4]) / 2) + ((0.0023682706810732711 * x[4] * pow(x[5], 2)) / 2) + ((0.01674088782171998 * x[2] * pow(x[4], 2)) / 2) + ((0.017880661999356982 * x[2] * pow(x[5], 2)) / 2) + ((0.031245422160302704 * pow(x[0], 2) * x[2]) / 2) + ((0.03168284188381567 * pow(x[1], 2) * x[2]) / 2) + ((0.056751613775557014 * x[0] * pow(x[2], 2)) / 2) + ((0.063318353145585149 * x[2] * pow(x[6], 2)) / 2) + ((0.52201577053917037 * pow(x[4], 3)) / 6) + ((1.6822708208525212 * pow(x[3], 2) * x[4]) / 2) + ((8.3696402244800971 * pow(x[2], 2) * x[6]) / 2) + ((9.8792685238541189 * pow(x[3], 2) * x[6]) / 2) + ((535.59780832409683 * x[2] * pow(x[3], 2)) / 2) + ((690.40982114572751 * pow(x[2], 3)) / 6)),
        #                  ( - 0.032090488704376238 * x[1] - 171.67340007243953 * x[3] - 7.1764113079695978e-06 * x[4] + 0.40410572991534277 * x[5] - 2.0090544711529489e-05 * x[6] - 7.1180776266778443 * x[7] + 9.7678247036896333e-06 * (x[0] * x[1] * x[2]) - 9.5668727061472977e-05 * (x[0] * x[1] * x[4]) + 0.00012970378400553901 * (x[0] * x[2] * x[3]) - 9.5950246360772129e-05 * (x[0] * x[4] * x[5]) + 0.00010813400937518589 * (x[0] * x[4] * x[7]) - 2.2198452209684476e-05 * (x[1] * x[2] * x[6]) + 0.00015274661523204623 * (x[1] * x[5] * x[7]) + 1.6778202130249511 * (x[2] * x[3] * x[4]) + 0.00056720485456160179 * (x[2] * x[3] * x[5]) + 9.8810740438233964 * (x[2] * x[3] * x[6]) + 2.0090544711529486e-05 * (x[2] * x[3] * x[7]) - 3.3094749262260714e-05 * (x[2] * x[4] * x[5]) - 0.016122381073757314 * (x[2] * x[5] * x[6]) + 2.0081325326653743 * (x[2] * x[6] * x[7]) + 0.017268648546613748 * (x[3] * x[5] * x[7]) + ((-0.021534870723141988 * pow(x[0], 2) * x[3]) / 2) + ((-0.02085033889723879 * pow(x[1], 2) * x[3]) / 2) + ((-0.019187387274015279 * x[3] * pow(x[5], 2)) / 2) + ((-0.017715055641039135 * x[3] * pow(x[4], 2)) / 2) + ((-0.0096633344683558862 * x[3] * pow(x[7], 2)) / 2) + ((-0.0043676840491391825 * x[1] * pow(x[2], 2)) / 2) + ((-0.003711361397315845 * x[1] * pow(x[5], 2)) / 2) + ((-0.00019590180721011275 * x[3] * pow(x[6], 2)) / 2) + ((-0.00013747195370884159 * x[1] * pow(x[7], 2)) / 2) + ((-9.7788249678741108e-05 * x[1] * pow(x[4], 2)) / 2) + ((-8.3032816197489804e-05 * pow(x[7], 3)) / 6) + ((-4.8312265534337242e-05 * pow(x[2], 2) * x[6]) / 2) + ((-1.4656049637581524e-05 * pow(x[0], 2) * x[1]) / 2) + ((-1.6940658945086007e-21 * pow(x[2], 2) * x[4]) / 2) + ((1.4352822615939194e-05 * pow(x[3], 2) * x[4]) / 2) + ((3.2049913311780707e-05 * pow(x[3], 2) * x[6]) / 2) + ((4.1537592663223022e-05 * pow(x[5], 2) * x[7]) / 2) + ((5.4874851266976834e-05 * x[5] * pow(x[7], 2)) / 2) + ((0.00010713558509562313 * pow(x[1], 2) * x[5]) / 2) + ((0.00034437205299724046 * pow(x[1], 3)) / 6) + ((0.0005095030685132694 * pow(x[0], 2) * x[5]) / 2) + ((0.0023682706810732707 * pow(x[4], 2) * x[5]) / 2) + ((0.0033359332990112733 * pow(x[2], 2) * x[7]) / 2) + ((0.020056114579288395 * x[1] * pow(x[3], 2)) / 2) + ((0.36322731298573557 * pow(x[2], 2) * x[5]) / 2) + ((0.59047199907046022 * pow(x[5], 3)) / 6) + ((1.5329323174055929 * pow(x[3], 2) * x[5]) / 2) + ((14.23417681605847 * pow(x[3], 2) * x[7]) / 2) + ((356.00989156174745 * pow(x[2], 2) * x[3]) / 2) + ((1267.887182536733 * pow(x[3], 3)) / 6))])
        Taylor3_Analytical_kp_10_k1k2_13 = np.array([x[4],
                         x[5],
                         x[6],
                         x[7],
                         (1.4113389626055488 * x[0] + 25.946924004825089 * x[2] + 0.10856453558504221 * x[4] + 1.0856453558504222 * x[6] + 1.7024595447594075 * (x[1] * x[2] * x[3]) + 0.13095842651995443 * (x[2] * x[3] * x[5]) + 1.3095842651995444 * (x[2] * x[3] * x[7]) + ((-2.1712907117008444 * x[2] * pow(x[6], 2)) / 2) + ((-2.1712907117008444 * x[2] * pow(x[7], 2)) / 2) + ((-7.5211396317510717e-17 * pow(x[3], 2) * x[4]) / 2) + ((0.2619168530399088 * pow(x[2], 2) * x[4]) / 2) + ((1.5335231745486666 * pow(x[2], 2) * x[6]) / 2) + ((3.4049190895188151 * x[0] * pow(x[2], 2)) / 2) + ((50.764593497768615 * x[2] * pow(x[3], 2)) / 2) + ((126.34685648848075 * pow(x[2], 3)) / 6)),
                         (1.4113389626055488 * x[1] + 25.946924004825089 * x[3] + 0.10856453558504221 * x[5] + 1.0856453558504222 * x[7] + 1.7024595447594075 * (x[0] * x[2] * x[3]) + 0.13095842651995443 * (x[2] * x[3] * x[4]) + 0.22393890934912206 * (x[2] * x[3] * x[6]) + ((-2.1712907117008444 * x[3] * pow(x[6], 2)) / 2) + ((-2.1712907117008444 * x[3] * pow(x[7], 2)) / 2) + ((-1.085645355850422 * pow(x[2], 2) * x[7]) / 2) + ((0.26191685303990886 * pow(x[3], 2) * x[5]) / 2) + ((1.5335231745486666 * pow(x[3], 2) * x[7]) / 2) + ((3.4049190895188151 * x[1] * pow(x[3], 2)) / 2) + ((24.81766949294353 * pow(x[2], 2) * x[3]) / 2) + ((126.34685648848075 * pow(x[3], 3)) / 6)),
                         ( - 1.5681544028950543 * x[0] - 17.929915560916761 * x[2] - 0.12062726176115801 * x[4] - 1.2062726176115801 * x[6] - 0.32346731350428737 * (x[1] * x[2] * x[3]) - 0.024882101038791349 * (x[2] * x[3] * x[5]) - 0.2488210103879136 * (x[2] * x[3] * x[7]) + ((-64.795649415561655 * pow(x[2], 3)) / 6) + ((-9.6452727645760437 * x[2] * pow(x[3], 2)) / 2) + ((-2.2150890299036288 * x[0] * pow(x[2], 2)) / 2) + ((-0.49764202077582698 * pow(x[2], 2) * x[6]) / 2) + ((-0.17039146383874065 * pow(x[2], 2) * x[4]) / 2) + ((8.3568218130567455e-17 * pow(x[3], 2) * x[4]) / 2) + ((0.41254523522316022 * x[2] * pow(x[7], 2)) / 2) + ((2.4125452352231602 * x[2] * pow(x[6], 2)) / 2)),
                         ( - 1.5681544028950543 * x[1] - 17.929915560916761 * x[3] - 0.12062726176115801 * x[5] - 1.2062726176115801 * x[7] - 1.8916217163993416 * (x[0] * x[2] * x[3]) - 0.14550936279994936 * (x[2] * x[3] * x[4]) - 0.24882101038791338 * (x[2] * x[3] * x[6]) + 2 * (x[2] * x[6] * x[7]) + ((-64.795649415561655 * pow(x[3], 3)) / 6) + ((-45.505103886409572 * pow(x[2], 2) * x[3]) / 2) + ((-2.2150890299036288 * x[1] * pow(x[3], 2)) / 2) + ((-1.5681544028950543 * x[1] * pow(x[2], 2)) / 2) + ((-0.49764202077582698 * pow(x[3], 2) * x[7]) / 2) + ((-0.17039146383874071 * pow(x[3], 2) * x[5]) / 2) + ((-0.12062726176115801 * pow(x[2], 2) * x[5]) / 2) + ((-4.4408920985006262e-16 * pow(x[2], 2) * x[7]) / 2) + ((2.4125452352231602 * x[3] * pow(x[6], 2)) / 2) + ((2.4125452352231602 * x[3] * pow(x[7], 2)) / 2))])
        return Taylor3_Analytical_kp_10_k1k2_13


def get(system_name):
    if isinstance(system_name, six.string_types):
        identifier = str(system_name)
        return globals()[identifier]()


def get_system(sys_name, deg_ftrs, deg_u, rm_one):
    system = get(sys_name)
    system.set_syms(deg_ftrs, deg_u, rm_one)
    return system


def get_monomials(x, deg, rm_one):
    c = 1 if isinstance(x[0], float) else Expression(1)
    _ = itertools.combinations_with_replacement(np.append(c, x), deg)
    basis = [np.prod(j) for j in _]
    if rm_one:
        basis = basis[1:]
    # if rm_one:
    #     print(np.array([i.ToExpression() for i in MonomialBasis(x, deg)[:-1]]))
    # else:
    #     print(np.array([i.ToExpression() for i in MonomialBasis(x, deg)]))
    # print(basis[::-1])
    return np.array(basis[:: -1])

    # def levelset_features(self, V, sigma_deg):
    #     self.sym_V = V
    #     self.sym_Vdot = self.sym_V.Jacobian(self.sym_x) @ self.sym_f
    #     self.degVdot = Polynomial(self.sym_Vdot, self.sym_x).TotalDegree()
    #     deg = int(np.floor((sigma_deg + self.degVdot - self.degV) / 2))
    #     self.sym_xxd = (self.sym_x.T@self.sym_x)**(deg)
    #     self.sym_sigma = get_monomials(self.sym_x, sigma_deg)
    #     psi_deg = int(np.floor(max(2 * deg + self.degV, sigma_deg +
    #                                self.degVdot) / 2))
    #     self.sym_psi = get_monomials(self.sym_x, psi_deg, rm_one)

    # def get_levelset_features(self, x):
    #     # x: (num_samples, sys_dim)
    #     n_samples = x.shape[0]
    #     V = np.zeros((n_samples, 1))
    #     Vdot = np.zeros((n_samples, 1))
    #     xxd = np.zeros((n_samples, 1))
    #     psi = np.zeros((n_samples, self.sym_psi.shape[0]))
    #     sigma = np.zeros((n_samples, self.sym_sigma.shape[0]))
    #     for i in range(n_samples):
    #         env = dict(zip(self.sym_x, x[i, :]))
    #         V[i, :] = self.sym_V.Evaluate(env)
    #         Vdot[i, :] = self.sym_Vdot.Evaluate(env)
    #         xxd[i, :] = self.sym_xxd.Evaluate(env)
    #         psi[i, :] = [i.Evaluate(env) for i in self.sym_psi]
    #         sigma[i, :] = [i.Evaluate(env) for i in self.sym_sigma]
    #     return [V, Vdot, xxd, psi, sigma]
