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
        # 3rd order taylor approximation  UNKNOWN CTRL
        # Taylor3_SDSOS_Q2e4_U_100 = np.array([1 * x[4],
        #                  1 * x [5],
        #                  1 * x[6],
        #                  1 * x[7],
        #                  (0.02131813480868517 * x[0] + 636.48541602386126 * x[2] + 1.2971001959678921 * x[4] - 0.00016722126725587293 * x[5] + 75.579654584175543 * x[6] + 0.0036512179130903121 * x[7] + 1.7295798484926599e-05 * (x[0] * x[1] * x[3]) + 0.0036092315348450177 * (x[0] * x[1] * x[5]) - 3.4992268914891108e-05 * (x[0] * x[1] * x[7]) - 0.0040192826539587516 * (x[0] * x[2] * x[4]) + 0.0040182183863080494 * (x[0] * x[2] * x[6]) - 0.0001625226371784559 * (x[0] * x[3] * x[4]) + 5.7565745006512755e-05 * (x[0] * x[3] * x[5]) + 0.0001462703734606103 * (x[0] * x[3] * x[6]) + 0.00046481151457350938 * (x[0] * x[3] * x[7]) - 0.0028611844719718732 * (x[1] * x[2] * x[3]) - 0.0039817649450264933 * (x[1] * x[2] * x[5]) - 2.2319926657905416e-05 * (x[1] * x[2] * x[7]) + 6.0646306868377566e-05 * (x[1] * x[3] * x[4]) - 0.0001614198312758099 * (x[1] * x[3] * x[5]) - 0.00061067978293268801 * (x[1] * x[3] * x[6]) - 0.0018599104799069349 * (x[1] * x[4] * x[5]) - 0.017367295863395377 * (x[2] * x[3] * x[4]) - 0.025353150067950372 * (x[2] * x[3] * x[5]) - 0.0068912369727216509 * (x[2] * x[3] * x[6]) - 0.0006923383635053657 * (x[2] * x[3] * x[7]) + 0.3732465496296995 * (x[2] * x[4] * x[6]) + 3.3723107559080209e-05 * (x[2] * x[5] * x[7]) - 1.9715687226880346e-05 * (x[3] * x[4] * x[5]) - 0.71802175130588775 * (x[3] * x[4] * x[7]) + 0.0009207807604353185 * (x[3] * x[5] * x[6]) + 0.0019840550386276772 * (x[3] * x[6] * x[7]) + 1.9957628207751896e-05 * (x[4] * x[5] * x[7]) + ((-876.1937761350988 * x[2] * pow(x[3], 2)) / 2) + ((-607.31991607355826 * pow(x[2], 3)) / 6) + ((-81.140716340953702 * pow(x[3], 2) * x[6]) / 2) + ((-11.450537637526194 * pow(x[3], 2) * x[4]) / 2) + ((-8.9961363618149015 * pow(x[4], 3)) / 6) + ((-1.1393974321088591 * x[2] * pow(x[6], 2)) / 2) + ((-1.0441732003965847 * pow(x[1], 2) * x[2]) / 2) + ((-0.99790431478960284 * pow(x[0], 2) * x[2]) / 2) + ((-0.91224557138273621 * x[4] * pow(x[5], 2)) / 2) + ((-0.87689413637162417 * x[2] * pow(x[7], 2)) / 2) + ((-0.74104066507172472 * x[0] * pow(x[2], 2)) / 2) + ((-0.41458143457628049 * x[2] * pow(x[5], 2)) / 2) + ((-0.41309005077199473 * x[2] * pow(x[4], 2)) / 2) + ((-0.17630862793651478 * pow(x[0], 3)) / 6) + ((-0.023129998173785998 * pow(x[1], 2) * x[4]) / 2) + ((-0.0027906707599179015 * x[0] * pow(x[4], 2)) / 2) + ((-0.0021065335172628925 * pow(x[3], 2) * x[7]) / 2) + ((-0.0018389708979083498 * x[0] * pow(x[5], 2)) / 2) + ((-0.00075025681097496961 * x[4] * pow(x[7], 2)) / 2) + ((5.5830610737882562e-06 * pow(x[2], 2) * x[5]) / 2) + ((0.00024434572886930463 * x[0] * pow(x[1], 2)) / 2) + ((0.00053772681777240793 * pow(x[3], 2) * x[5]) / 2) + ((0.0016956773898039153 * pow(x[2], 2) * x[7]) / 2) + ((0.055999526690730161 * x[0] * pow(x[3], 2)) / 2) + ((0.086646908040072551 * pow(x[0], 2) * x[4]) / 2) + ((1.5948301058326182 * pow(x[2], 2) * x[6]) / 2) + ((4.851380771952833 * pow(x[2], 2) * x[4]) / 2)),
        #                  (0.010436819485440268 * x[1] + 630.56574963994115 * x[3] - 0.00033461800469045409 * x[4] - 1.2653618992734783 * x[5] + 0.0035005608493991888 * x[6] + 53.440821826969824 * x[7] - 1.6224521102067183e-05 * (x[0] * x[1] * x[2]) + 0.003648111833639341 * (x[0] * x[1] * x[4]) - 0.011256145836253782 * (x[0] * x[2] * x[3]) - 2.8136799107067071e-05 * (x[0] * x[2] * x[5]) + 2.7724505130501521e-05 * (x[0] * x[2] * x[7]) - 0.00043881079743200772 * (x[0] * x[3] * x[4]) + 1.8072626378840959e-05 * (x[0] * x[3] * x[6]) - 0.0018389708979083498 * (x[0] * x[4] * x[5]) + 2.3129581245703778e-05 * (x[1] * x[2] * x[4]) + 0.0035820851469870714 * (x[1] * x[2] * x[6]) - 0.00039888296745561652 * (x[1] * x[3] * x[5]) + 0.00014527784814822889 * (x[1] * x[3] * x[6]) + 0.00048256588170695139 * (x[1] * x[3] * x[7]) - 7.8674215086334662 * (x[2] * x[3] * x[4]) - 0.018233379347251468 * (x[2] * x[3] * x[5]) - 82.638208445247997 * (x[2] * x[3] * x[6]) - 0.0032439927436789338 * (x[2] * x[3] * x[7]) + 0.00074732294079478368 * (x[2] * x[4] * x[5]) + 1.4608279570712771e-05 * (x[2] * x[4] * x[7]) + 0.37382960487292677 * (x[2] * x[5] * x[6]) - 0.87691133971681356 * (x[2] * x[6] * x[7]) + 5.0435567437498079e-05 * (x[3] * x[4] * x[6]) - 0.53246493510151294 * (x[3] * x[5] * x[7]) + ((-2531.1088815758362 * pow(x[3], 3)) / 6) + ((-1644.266501494375 * pow(x[2], 2) * x[3]) / 2) + ((-53.803663530289121 * pow(x[2], 2) * x[7]) / 2) + ((-53.545596896935905 * pow(x[3], 2) * x[7]) / 2) + ((-9.8493794401427728 * pow(x[5], 3)) / 6) + ((-4.3797850612261797 * pow(x[3], 2) * x[5]) / 2) + ((-0.91226774652518927 * pow(x[4], 2) * x[5]) / 2) + ((-0.74612722414349664 * x[1] * pow(x[2], 2)) / 2) + ((-0.16056224511248149 * pow(x[1], 3)) / 6) + ((-0.013696140921972503 * pow(x[0], 2) * x[5]) / 2) + ((-0.0029670443804646895 * x[1] * pow(x[5], 2)) / 2) + ((-0.002355866302916182 * pow(x[3], 2) * x[6]) / 2) + ((-0.0018599104799069349 * x[1] * pow(x[4], 2)) / 2) + ((-0.0009809713004056168 * x[5] * pow(x[7], 2)) / 2) + ((-8.7033382841245595e-05 * pow(x[2], 2) * x[4]) / 2) + ((0.0001960485470849441 * pow(x[0], 2) * x[1]) / 2) + ((0.00041032189574753089 * pow(x[7], 3)) / 6) + ((0.00042808712670333528 * pow(x[3], 2) * x[4]) / 2) + ((0.00051166330601521185 * pow(x[0], 2) * x[7]) / 2) + ((0.00083558887410661647 * pow(x[5], 2) * x[7]) / 2) + ((0.00085357630706882923 * pow(x[4], 2) * x[7]) / 2) + ((0.00096846110568437412 * pow(x[1], 2) * x[7]) / 2) + ((0.0012007443649296391 * x[3] * pow(x[6], 2)) / 2) + ((0.0052635405033706964 * pow(x[2], 2) * x[6]) / 2) + ((0.051111096093520526 * x[1] * pow(x[3], 2)) / 2) + ((0.08962619098926336 * pow(x[1], 2) * x[5]) / 2) + ((0.20662845524292764 * x[3] * pow(x[7], 2)) / 2) + ((0.5646264743366618 * pow(x[2], 2) * x[5]) / 2) + ((0.59150628176786513 * x[3] * pow(x[5], 2)) / 2) + ((0.79783826972757899 * x[3] * pow(x[4], 2)) / 2) + ((0.94501943941732536 * pow(x[0], 2) * x[3]) / 2) + ((1.0235302304680154 * pow(x[1], 2) * x[3]) / 2)),
        #                  ( - 0.023686816454094633 * x[0] - 696.30601780429026 * x[2] - 1.4412224399643245 * x[4] + 0.00018580140806208101 * x[5] - 83.977393982417269 * x[6] - 0.0040569087923225684 * x[7] - 1.9217553872140666e-05 * (x[0] * x[1] * x[3]) - 0.0040102572609389082 * (x[0] * x[1] * x[5]) + 3.8880298794323451e-05 * (x[0] * x[1] * x[7]) + 0.0044658696155097238 * (x[0] * x[2] * x[4]) - 0.0044646870958978324 * (x[0] * x[2] * x[6]) + 0.00018058070797606206 * (x[0] * x[3] * x[4]) - 6.3961938896125282e-05 * (x[0] * x[3] * x[5]) - 0.0001625226371784559 * (x[0] * x[3] * x[6]) - 0.0005164572384150104 * (x[0] * x[3] * x[7]) + 0.014775559952680158 * (x[1] * x[2] * x[3]) + 0.0044241832722516587 * (x[1] * x[2] * x[5]) + 2.4799918508783796e-05 * (x[1] * x[2] * x[7]) - 6.7384785409308409e-05 * (x[1] * x[3] * x[4]) + 0.00017935536808423323 * (x[1] * x[3] * x[5]) + 0.00067853309214743101 * (x[1] * x[3] * x[6]) + 0.0020665671998965941 * (x[1] * x[4] * x[5]) + 0.018925197620783249 * (x[2] * x[3] * x[4]) - 1.3777874991172532 * (x[2] * x[3] * x[5]) + 0.011546442024578709 * (x[2] * x[3] * x[6]) + 59.379460183703699 * (x[2] * x[3] * x[7]) - 0.41471838847744386 * (x[2] * x[4] * x[6]) - 3.7470119510089113e-05 * (x[2] * x[5] * x[7]) + 2.1906319140978161e-05 * (x[3] * x[4] * x[5]) + 0.79780194589543074 * (x[3] * x[4] * x[7]) - 0.0010230897338170206 * (x[3] * x[5] * x[6]) - 0.0022045055984751968 * (x[3] * x[6] * x[7]) - 2.2175142453057661e-05 * (x[4] * x[5] * x[7]) + ((-3.9492006399832666 * pow(x[2], 2) * x[4]) / 2) + ((-1.0256731818093066 * x[2] * pow(x[7], 2)) / 2) + ((-0.096274342266747276 * pow(x[0], 2) * x[4]) / 2) + ((-0.062221696323033503 * x[0] * pow(x[3], 2)) / 2) + ((-0.0005974742419693421 * pow(x[3], 2) * x[5]) / 2) + ((-0.00027149525429922731 * x[0] * pow(x[1], 2)) / 2) + ((-0.00019200480925517908 * pow(x[2], 2) * x[5]) / 2) + ((0.00083361867886107722 * x[4] * pow(x[7], 2)) / 2) + ((0.0020433009976759441 * x[0] * pow(x[5], 2)) / 2) + ((0.0021728228036515516 * pow(x[2], 2) * x[7]) / 2) + ((0.0023405927969587692 * pow(x[3], 2) * x[7]) / 2) + ((0.0031007452887976683 * x[0] * pow(x[4], 2)) / 2) + ((0.025699997970873332 * pow(x[1], 2) * x[4]) / 2) + ((0.1958984754850164 * pow(x[0], 3)) / 6) + ((0.45898894530221634 * x[2] * pow(x[4], 2)) / 2) + ((0.46064603841808943 * x[2] * pow(x[5], 2)) / 2) + ((0.84706533320045541 * x[0] * pow(x[2], 2)) / 2) + ((1.0136061904252625 * x[4] * pow(x[5], 2)) / 2) + ((1.1087825719884477 * pow(x[0], 2) * x[2]) / 2) + ((1.160192444885094 * pow(x[1], 2) * x[2]) / 2) + ((1.2659971467876212 * x[2] * pow(x[6], 2)) / 2) + ((9.9957070686832239 * pow(x[4], 3)) / 6) + ((12.722819597251327 * pow(x[3], 2) * x[4]) / 2) + ((82.205360531492133 * pow(x[2], 2) * x[6]) / 2) + ((90.156351489948563 * pow(x[3], 2) * x[6]) / 2) + ((2363.905861572201 * x[2] * pow(x[3], 2)) / 2) + ((2785.5179601612685 * pow(x[2], 3)) / 6)),
        #                  ( - 0.011596466094933632 * x[1] - 689.72861071104569 * x[3] + 0.00037179778298939342 * x[4] + 1.4059576658594202 * x[5] - 0.0038895120548879873 * x[6] - 59.378690918855362 * x[7] + 1.8027245668963537e-05 * (x[0] * x[1] * x[2]) - 0.0040534575929326005 * (x[0] * x[1] * x[4]) + 0.012506828706948645 * (x[0] * x[2] * x[3]) + 3.1263110118963409e-05 * (x[0] * x[2] * x[5]) - 3.0805005700557242e-05 * (x[0] * x[2] * x[7]) + 0.00048756755270223077 * (x[0] * x[3] * x[4]) - 2.0080695976489955e-05 * (x[0] * x[3] * x[6]) + 0.0020433009976759441 * (x[0] * x[4] * x[5]) - 2.5699534717448639e-05 * (x[1] * x[2] * x[4]) - 0.0039800946077634125 * (x[1] * x[2] * x[6]) + 0.00044320329717290725 * (x[1] * x[3] * x[5]) - 0.00016141983127580985 * (x[1] * x[3] * x[6]) - 0.00053618431300772368 * (x[1] * x[3] * x[7]) + 8.7415794540371845 * (x[2] * x[3] * x[4]) + 0.020259310385834964 * (x[2] * x[3] * x[5]) + 91.820231605831097 * (x[2] * x[3] * x[6]) + 0.0036044363818654819 * (x[2] * x[3] * x[7]) - 0.00083035882310531514 * (x[2] * x[4] * x[5]) - 1.6231421745236412e-05 * (x[2] * x[4] * x[7]) - 0.41536622763658526 * (x[2] * x[5] * x[6]) + 2.9743459330186814 * (x[2] * x[6] * x[7]) - 5.603951937499786e-05 * (x[3] * x[4] * x[6]) + 0.59162770566834766 * (x[3] * x[5] * x[7]) + ((-1.1372558116311282 * pow(x[1], 2) * x[3]) / 2) + ((-1.0500215993525837 * pow(x[0], 2) * x[3]) / 2) + ((-0.88648696636397661 * x[3] * pow(x[4], 2)) / 2) + ((-0.65722920196429446 * x[3] * pow(x[5], 2)) / 2) + ((-0.22958717249214181 * x[3] * pow(x[7], 2)) / 2) + ((-0.099584656654737064 * pow(x[1], 2) * x[5]) / 2) + ((-0.045193640675644728 * x[1] * pow(x[3], 2)) / 2) + ((-0.0097378903919665381 * pow(x[2], 2) * x[6]) / 2) + ((-0.0013341604054773768 * x[3] * pow(x[6], 2)) / 2) + ((-0.0010760678952048601 * pow(x[1], 2) * x[7]) / 2) + ((-0.0009484181189653658 * pow(x[4], 2) * x[7]) / 2) + ((-0.00092843208234068492 * pow(x[5], 2) * x[7]) / 2) + ((-0.00084745014599309925 * pow(x[3], 2) * x[4]) / 2) + ((-0.00056851478446134636 * pow(x[0], 2) * x[7]) / 2) + ((-0.00045591321749725649 * pow(x[7], 3)) / 6) + ((-0.0002178317189832712 * pow(x[0], 2) * x[1]) / 2) + ((0.0004685015417018885 * pow(x[2], 2) * x[4]) / 2) + ((0.0010899681115617962 * x[5] * pow(x[7], 2)) / 2) + ((0.0020665671998965941 * x[1] * pow(x[4], 2)) / 2) + ((0.0032967159782940989 * x[1] * pow(x[5], 2)) / 2) + ((0.0065071412803504123 * pow(x[3], 2) * x[6]) / 2) + ((0.015217934357747225 * pow(x[0], 2) * x[5]) / 2) + ((0.17840249456942386 * pow(x[1], 3)) / 6) + ((0.40315744813254639 * pow(x[2], 2) * x[7]) / 2) + ((0.77859491659646274 * pow(x[2], 2) * x[5]) / 2) + ((0.81743378295339586 * x[1] * pow(x[2], 2)) / 2) + ((1.0136308294724325 * pow(x[4], 2) * x[5]) / 2) + ((3.4604701799474453 * pow(x[3], 2) * x[5]) / 2) + ((10.943754933491968 * pow(x[5], 3)) / 6) + ((118.87379858211747 * pow(x[3], 2) * x[7]) / 2) + ((1137.2341687271485 * pow(x[2], 2) * x[3]) / 2) + ((4903.329033884067 * pow(x[3], 3)) / 6))])
        # # nonlinear system equations  UNKNOWN CTRL
        # Full_SDSOS_Q2e4_U100 = np.array([x[4],
        #                  x[5],
        #                  x[6],
        #                  x[7],
        #                  (-0.5 * (10 * ( - 0.0070255632221867936 * x[0] + 0.43274946929264047 * x[4] + 3.5370873498777211e-05 * x[5] + 1.0684296952698988 * x[6] - 2.1919220402320487e-05 * x[7] - 0.00072184630696900355 * (x[0] * x[1] * x[5]) + 6.9984537829782217e-06 * (x[0] * x[1] * x[7]) - 3.4591596969853197e-06 * (x[0] * x[1] * sin(x[3])) - 2.4434572886930464e-05 * (x[0] * pow(x[1], 2)) - 1.5935558915818625e-05 * (x[0] * x[4] * sin(x[2])) + 0.00027906707599179015 * (x[0] * pow(x[4], 2)) - 7.8968118646176109e-06 * (x[0] * x[5] * sin(x[3])) + 0.00018389708979083498 * (x[0] * pow(x[5], 2)) - 0.00073781288073681195 * (x[0] * x[6] * sin(x[2])) - 2.925407469212206e-05 * (x[0] * x[6] * sin(x[3])) - 8.6093074215917243e-05 * (x[0] * x[7] * sin(x[3])) + 0.12218739661935774 * (x[0] * pow(sin(x[2]), 2)) - 0.013393504249813403 * (x[0] * pow(sin(x[3]), 2)) - 0.091939144189822478 * (x[0] * cos(x[2])) + 0.00014211435304352873 * (x[0] * cos(x[2]) * cos(x[3])) + 0.094449350252944445 * (x[0] * pow(cos(x[2]), 2)) + 0.016720496219958907 * (x[0] * cos(x[3])) - 0.014351714648301811 * (x[0] * pow(cos(x[3]), 2)) - 0.0086646908040072558 * (pow(x[0], 2) * x[4]) - 3.3614392503734857e-05 * (pow(x[0], 2) * sin(x[2])) + 0.00037198209598138699 * (x[1] * x[4] * x[5]) - 8.6879398306825367e-06 * (x[1] * x[4] * sin(x[3])) - 4.6259162491407557e-06 * (x[1] * x[5] * sin(x[2])) + 3.0971893886936787e-06 * (x[1] * x[6] * sin(x[3])) - 4.1276919764238341e-06 * (x[1] * sin(x[2]) * sin(x[3])) + 0.0023129998173785997 * (pow(x[1], 2) * x[4]) - 3.5114424889291614e-05 * (pow(x[1], 2) * sin(x[2])) - 3.9915256415503791e-06 * (x[4] * x[5] * x[7]) + 5.4130420663419832e-06 * (x[4] * x[5] * sin(x[3])) + 0.091224557138273626 * (x[4] * pow(x[5], 2)) - 0.07468123277600737 * (x[4] * x[6] * sin(x[2])) + 0.14361576028882392 * (x[4] * x[7] * sin(x[3])) + 7.5025681097496959e-05 * (x[4] * pow(x[7], 2)) + 0.0034898095519765031 * (x[4] * sin(x[2]) * sin(x[3])) + 0.066569319400862345 * (x[4] * pow(sin(x[2]), 2)) + 0.36528825942112469 * (x[4] * pow(sin(x[3]), 2)) - 0.086333871747806323 * (x[4] * cos(x[2])) - 0.043069557043361448 * (x[4] * cos(x[2]) * cos(x[3])) + 0.79505652787013004 * (x[4] * pow(cos(x[2]), 2)) + 0.0033188004827935914 * (x[4] * cos(x[3])) + 0.21278195531873381 * (x[4] * pow(cos(x[3]), 2)) - 0.00018056868724906764 * (pow(x[4], 2) * sin(x[2])) + 1.3229141588693226e-06 * (x[5] * x[6] * sin(x[3])) - 0.00046638731657011118 * (x[5] * sin(x[2]) * sin(x[3])) - 3.5605819390869218e-06 * (x[5] * pow(sin(x[2]), 2)) - 2.2632939944443235e-05 * (x[5] * pow(sin(x[3]), 2)) - 4.1751174868883975e-05 * (x[5] * cos(x[2]) * cos(x[3])) - 3.0836430809464263e-06 * (x[5] * pow(cos(x[2]), 2)) + 1.5448073387723312e-05 * (x[5] * cos(x[3])) - 2.3479288291093966e-05 * (x[5] * pow(cos(x[3]), 2)) - 7.4732294079478368e-05 * (pow(x[5], 2) * sin(x[2])) + 1.0269024881732043e-05 * (x[6] * x[7] * sin(x[3])) + 1.4715341367685285e-05 * (x[6] * sin(x[2]) * sin(x[3])) + 8.9228799027727701e-06 * (x[6] * pow(sin(x[2]), 2)) + 0.0014043403211331732 * (x[6] * pow(sin(x[3]), 2)) + 0.013280304365897614 * (x[6] * cos(x[2])) + 0.3083028917750289 * (x[6] * cos(x[2]) * cos(x[3])) + 0.024251762935323457 * (x[6] * cos(x[3])) + 0.0022663726838885426 * (x[6] * pow(cos(x[3]), 2)) - 1.4365282530359833e-05 * (pow(x[6], 2) * sin(x[2])) - 2.4593108899959271e-05 * (x[7] * sin(x[2]) * sin(x[3])) + 1.2343239101463873e-05 * (x[7] * pow(sin(x[2]), 2)) + 1.6336381033981899e-06 * (x[7] * pow(sin(x[3]), 2)) - 1.124897285814006e-05 * (x[7] * cos(x[2])) - 1.9368052393821935e-05 * (x[7] * cos(x[2]) * cos(x[3])) - 2.0862469667145969e-05 * (x[7] * pow(cos(x[2]), 2)) - 1.5166495839347087e-06 * (x[7] * cos(x[3])) - 1.0615193579000283e-06 * (x[7] * pow(cos(x[3]), 2)) + 1.3147451613641493e-06 * (pow(x[7], 2) * sin(x[2])) - 0.0032281880387797545 * (sin(x[2]) * pow(sin(x[3]), 2)) + 0.00029121976419156144 * (sin(x[2]) * cos(x[2])) + 0.00013129780089208464 * (sin(x[2]) * cos(x[2]) * cos(x[3])) - 0.00029945293048839322 * (sin(x[2]) * pow(cos(x[2]), 2)) + 0.00054758808043536844 * (sin(x[2]) * cos(x[3])) - 0.0028217641589934508 * (sin(x[2]) * pow(cos(x[3]), 2)) + 0.0058769542645504928 * pow(x[0], 3) + 0.29987121206049672 * pow(x[4], 3) - 0.00034017094657775996 * pow(sin(x[2]), 3) - 0.002076395707890478 * sin(x[2])) + 10 * ((0.00038532003574531912 * x[0] + 1.0684296952698988 * x[4] + 3.2580943371291799e-05 * x[5] + 14.806257609508169 * x[6] + 0.00095370464881419312 * x[7] - 0.00073781288073681195 * (x[0] * x[4] * sin(x[2])) - 2.925407469212206e-05 * (x[0] * x[4] * sin(x[3])) + 3.2547034230164453e-06 * (x[0] * x[5] * sin(x[3])) + 5.924771687231811e-05 * (x[0] * x[6] * sin(x[2])) + 6.1823058289061733e-06 * (x[0] * x[7] * sin(x[3])) - 2.1319484037325249e-06 * (x[0] * pow(sin(x[2]), 2)) + 7.4652077840916482e-05 * (x[0] * pow(sin(x[3]), 2)) + 0.00032039783441772897 * (x[0] * cos(x[2])) + 0.00051684766904027204 * (x[0] * cos(x[2]) * cos(x[3])) + 0.00065050390476681819 * (x[0] * cos(x[3])) + 0.00016017971066541138 * (x[0] * pow(cos(x[3]), 2)) - 0.089841641284317611 * (pow(x[0], 2) * sin(x[2])) + 3.0971893886936787e-06 * (x[1] * x[4] * sin(x[3])) - 0.00072088101472899541 * (x[1] * x[5] * sin(x[2])) - 2.9055569629645779e-05 * (x[1] * x[5] * sin(x[3])) - 0.00010713489047805953 * (x[1] * x[6] * sin(x[3])) - 4.0175867984229744e-06 * (x[1] * x[7] * sin(x[2])) - 0.00051872812773371866 * (x[1] * sin(x[2]) * sin(x[3])) - 0.094007191018092975 * (pow(x[1], 2) * sin(x[2])) + 1.3229141588693226e-06 * (x[4] * x[5] * sin(x[3])) - 2.8730565060719665e-05 * (x[4] * x[6] * sin(x[2])) + 1.0269024881732043e-05 * (x[4] * x[7] * sin(x[3])) + 1.4715341367685285e-05 * (x[4] * sin(x[2]) * sin(x[3])) + 8.9228799027727701e-06 * (x[4] * pow(sin(x[2]), 2)) + 0.0014043403211331732 * (x[4] * pow(sin(x[3]), 2)) + 0.013280304365897614 * (x[4] * cos(x[2])) + 0.3083028917750289 * (x[4] * cos(x[2]) * cos(x[3])) + 0.024251762935323457 * (x[4] * cos(x[3])) + 0.0022663726838885426 * (x[4] * pow(cos(x[3]), 2)) - 0.037340616388003685 * (pow(x[4], 2) * sin(x[2])) + 0.00016693115962133971 * (x[5] * x[6] * sin(x[3])) + 6.0701593606344377e-06 * (x[5] * x[7] * sin(x[2])) - 0.0049833155971441669 * (x[5] * sin(x[2]) * sin(x[3])) - 6.3636166956642285e-06 * (x[5] * pow(sin(x[3]), 2)) - 3.772251843062457e-05 * (x[5] * cos(x[2]) * cos(x[3])) - 3.0261503939696959e-05 * (x[5] * cos(x[3])) - 1.0442392526008848e-05 * (x[5] * pow(cos(x[3]), 2)) - 0.03737958817653677 * (pow(x[5], 2) * sin(x[2])) + 0.00036637202934654078 * (x[6] * x[7] * sin(x[3])) - 0.0012271788478589803 * (x[6] * sin(x[2]) * sin(x[3])) - 0.00036033240030784372 * (x[6] * pow(sin(x[2]), 2)) + 0.059746490847326778 * (x[6] * pow(sin(x[3]), 2)) - 7.5040755460096105e-05 * (x[6] * cos(x[2])) + 0.0016937722479819658 * (x[6] * cos(x[2]) * cos(x[3])) - 0.0020844318761020636 * (x[6] * cos(x[3])) + 0.073423840354042841 * (x[6] * pow(cos(x[3]), 2)) - 0.10255869764407463 * (pow(x[6], 2) * sin(x[2])) - 0.00014675470344092915 * (x[7] * sin(x[2]) * sin(x[3])) + 2.4143350725201305e-05 * (x[7] * pow(sin(x[2]), 2)) + 1.0891086859192866e-05 * (x[7] * pow(sin(x[3]), 2)) - 0.00018459355870316878 * (x[7] * cos(x[2])) - 0.0001596672685213968 * (x[7] * cos(x[2]) * cos(x[3])) + 1.0654482241917294e-05 * (x[7] * cos(x[3])) - 3.1258275312225509e-05 * (x[7] * pow(cos(x[3]), 2)) - 0.078919289002800944 * (pow(x[7], 2) * sin(x[2])) - 0.4321707452378572 * (sin(x[2]) * pow(sin(x[3]), 2)) + 1.5564019369976865 * (sin(x[2]) * cos(x[2])) - 0.73547405826909218 * (sin(x[2]) * cos(x[2]) * cos(x[3])) - 1.5352900739413957 * (sin(x[2]) * pow(cos(x[2]), 2)) + 42.984364896010824 * (sin(x[2]) * cos(x[3])) + 0.019709904259417082 * (sin(x[2]) * pow(cos(x[3]), 2)) - 0.25008791518677898 * pow(sin(x[2]), 3) + 72.27385752280091 * sin(x[2])) * ((-1 * cos(x[3])) / 0.90000000000000002)))),
        #                  (-0.5 * (10 * ( - 0.0065597048649271477 * x[1] + 3.5370873498777211e-05 * x[4] + 0.18283061060819256 * x[5] + 3.2580943371291799e-05 * x[6] - 0.0063134920963305342 * x[7] - 0.00072184630696900355 * (x[0] * x[1] * x[4]) + 3.2449042204134365e-06 * (x[0] * x[1] * sin(x[2])) + 0.00036779417958166996 * (x[0] * x[4] * x[5]) - 7.8968118646176109e-06 * (x[0] * x[4] * sin(x[3])) + 5.6273598214134143e-06 * (x[0] * x[5] * sin(x[2])) + 3.2547034230164453e-06 * (x[0] * x[6] * sin(x[3])) - 1.0186598971223059e-05 * (x[0] * sin(x[2]) * sin(x[3])) - 2.8594868097985197e-05 * (pow(x[0], 2) * x[1]) + 0.0012409276895305038 * (pow(x[0], 2) * x[5]) - 0.00011581776240007176 * (pow(x[0], 2) * x[7]) + 3.7192594459799988e-05 * (pow(x[0], 2) * sin(x[3])) - 4.6259162491407557e-06 * (x[1] * x[4] * sin(x[2])) + 0.00018599104799069349 * (x[1] * pow(x[4], 2)) - 2.0707620350439519e-05 * (x[1] * x[5] * sin(x[3])) + 0.00029670443804646895 * (x[1] * pow(x[5], 2)) - 0.00072088101472899541 * (x[1] * x[6] * sin(x[2])) - 2.9055569629645779e-05 * (x[1] * x[6] * sin(x[3])) - 9.0435792457406537e-05 * (x[1] * x[7] * sin(x[3])) + 0.12156527713545084 * (x[1] * pow(sin(x[2]), 2)) - 0.011546315046333411 * (x[1] * pow(sin(x[3]), 2)) - 0.087966473726447128 * (x[1] * cos(x[2])) + 0.0014347684052675109 * (x[1] * cos(x[2]) * cos(x[3])) + 0.090639885085965918 * (x[1] * pow(cos(x[2]), 2)) + 0.015068584266482114 * (x[1] * cos(x[3])) - 0.01245848954964501 * (x[1] * pow(cos(x[3]), 2)) - 0.0090866721254703152 * (pow(x[1], 2) * x[5]) - 0.00011164772388958151 * (pow(x[1], 2) * x[7]) + 3.3894269510030535e-05 * (pow(x[1], 2) * sin(x[3])) - 0.00014946458815895674 * (x[4] * x[5] * sin(x[2])) + 1.3229141588693226e-06 * (x[4] * x[6] * sin(x[3])) - 0.00046638731657011118 * (x[4] * sin(x[2]) * sin(x[3])) - 3.5605819390869218e-06 * (x[4] * pow(sin(x[2]), 2)) - 2.2632939944443235e-05 * (x[4] * pow(sin(x[3]), 2)) - 4.1751174868883975e-05 * (x[4] * cos(x[2]) * cos(x[3])) - 3.0836430809464263e-06 * (x[4] * pow(cos(x[2]), 2)) + 1.5448073387723312e-05 * (x[4] * cos(x[3])) - 2.3479288291093966e-05 * (x[4] * pow(cos(x[3]), 2)) + 0.091224557138273626 * (pow(x[4], 2) * x[5]) - 1.9957628207751896e-06 * (pow(x[4], 2) * x[7]) + 2.7065210331709916e-06 * (pow(x[4], 2) * sin(x[3])) - 0.074759176353073539 * (x[5] * x[6] * sin(x[2])) + 0.10649298702030259 * (x[5] * x[7] * sin(x[3])) + 7.2100335269445267e-05 * (x[5] * pow(x[7], 2)) + 0.0036976152822558923 * (x[5] * sin(x[2]) * sin(x[3])) + 0.074265982189716306 * (x[5] * pow(sin(x[2]), 2)) + 0.45834448690085838 * (x[5] * pow(sin(x[3]), 2)) - 0.14778765323579626 * (x[5] * cos(x[2])) - 0.045042136635330805 * (x[5] * cos(x[2]) * cos(x[3])) + 0.222497791565338 * (x[5] * pow(cos(x[2]), 2)) - 0.0030011200314892816 * (x[5] * cos(x[3])) + 0.31764079677561585 * (x[5] * pow(cos(x[3]), 2)) - 3.4474037779446669e-06 * (pow(x[5], 2) * x[7]) + 1.2142390048262233e-05 * (pow(x[5], 2) * sin(x[3])) + 6.0701593606344377e-06 * (x[6] * x[7] * sin(x[2])) - 0.0049833155971441669 * (x[6] * sin(x[2]) * sin(x[3])) - 6.3636166956642285e-06 * (x[6] * pow(sin(x[3]), 2)) - 3.772251843062457e-05 * (x[6] * cos(x[2]) * cos(x[3])) - 3.0261503939696959e-05 * (x[6] * cos(x[3])) - 1.0442392526008848e-05 * (x[6] * pow(cos(x[3]), 2)) + 8.3465579810669855e-05 * (pow(x[6], 2) * sin(x[3])) + 0.011803064562443325 * (x[7] * pow(sin(x[2]), 2)) + 0.00025590203268803128 * (x[7] * pow(sin(x[3]), 2)) - 0.0012586343227540559 * (x[7] * cos(x[2])) + 0.24432941235194164 * (x[7] * cos(x[2]) * cos(x[3])) + 0.0094161749899277763 * (x[7] * pow(cos(x[2]), 2)) - 0.00040709653618897779 * (x[7] * cos(x[3])) + 0.00089295388605508286 * (x[7] * pow(cos(x[3]), 2)) - 0.0030776700047295234 * (pow(sin(x[2]), 2) * sin(x[3])) - 0.0056215135530187773 * (sin(x[3]) * cos(x[2])) + 0.023072964768245035 * (sin(x[3]) * cos(x[2]) * cos(x[3])) - 0.0025689842009279191 * (sin(x[3]) * pow(cos(x[2]), 2)) - 0.0023865963238517305 * (sin(x[3]) * cos(x[3])) + 0.010998779258773987 * (sin(x[3]) * pow(cos(x[3]), 2)) + 0.0053441513135730721 * pow(x[1], 3) + 0.32831137118854503 * pow(x[5], 3) - 7.7990384313349214e-06 * pow(x[7], 3) + 0.0071561515685901571 * pow(sin(x[3]), 3) - 0.029147178994592283 * sin(x[3])) + 10 * ((0.00038532003574531912 * x[0] + 1.0684296952698988 * x[4] + 3.2580943371291799e-05 * x[5] + 14.806257609508169 * x[6] + 0.00095370464881419312 * x[7] - 0.00073781288073681195 * (x[0] * x[4] * sin(x[2])) - 2.925407469212206e-05 * (x[0] * x[4] * sin(x[3])) + 3.2547034230164453e-06 * (x[0] * x[5] * sin(x[3])) + 5.924771687231811e-05 * (x[0] * x[6] * sin(x[2])) + 6.1823058289061733e-06 * (x[0] * x[7] * sin(x[3])) - 2.1319484037325249e-06 * (x[0] * pow(sin(x[2]), 2)) + 7.4652077840916482e-05 * (x[0] * pow(sin(x[3]), 2)) + 0.00032039783441772897 * (x[0] * cos(x[2])) + 0.00051684766904027204 * (x[0] * cos(x[2]) * cos(x[3])) + 0.00065050390476681819 * (x[0] * cos(x[3])) + 0.00016017971066541138 * (x[0] * pow(cos(x[3]), 2)) - 0.089841641284317611 * (pow(x[0], 2) * sin(x[2])) + 3.0971893886936787e-06 * (x[1] * x[4] * sin(x[3])) - 0.00072088101472899541 * (x[1] * x[5] * sin(x[2])) - 2.9055569629645779e-05 * (x[1] * x[5] * sin(x[3])) - 0.00010713489047805953 * (x[1] * x[6] * sin(x[3])) - 4.0175867984229744e-06 * (x[1] * x[7] * sin(x[2])) - 0.00051872812773371866 * (x[1] * sin(x[2]) * sin(x[3])) - 0.094007191018092975 * (pow(x[1], 2) * sin(x[2])) + 1.3229141588693226e-06 * (x[4] * x[5] * sin(x[3])) - 2.8730565060719665e-05 * (x[4] * x[6] * sin(x[2])) + 1.0269024881732043e-05 * (x[4] * x[7] * sin(x[3])) + 1.4715341367685285e-05 * (x[4] * sin(x[2]) * sin(x[3])) + 8.9228799027727701e-06 * (x[4] * pow(sin(x[2]), 2)) + 0.0014043403211331732 * (x[4] * pow(sin(x[3]), 2)) + 0.013280304365897614 * (x[4] * cos(x[2])) + 0.3083028917750289 * (x[4] * cos(x[2]) * cos(x[3])) + 0.024251762935323457 * (x[4] * cos(x[3])) + 0.0022663726838885426 * (x[4] * pow(cos(x[3]), 2)) - 0.037340616388003685 * (pow(x[4], 2) * sin(x[2])) + 0.00016693115962133971 * (x[5] * x[6] * sin(x[3])) + 6.0701593606344377e-06 * (x[5] * x[7] * sin(x[2])) - 0.0049833155971441669 * (x[5] * sin(x[2]) * sin(x[3])) - 6.3636166956642285e-06 * (x[5] * pow(sin(x[3]), 2)) - 3.772251843062457e-05 * (x[5] * cos(x[2]) * cos(x[3])) - 3.0261503939696959e-05 * (x[5] * cos(x[3])) - 1.0442392526008848e-05 * (x[5] * pow(cos(x[3]), 2)) - 0.03737958817653677 * (pow(x[5], 2) * sin(x[2])) + 0.00036637202934654078 * (x[6] * x[7] * sin(x[3])) - 0.0012271788478589803 * (x[6] * sin(x[2]) * sin(x[3])) - 0.00036033240030784372 * (x[6] * pow(sin(x[2]), 2)) + 0.059746490847326778 * (x[6] * pow(sin(x[3]), 2)) - 7.5040755460096105e-05 * (x[6] * cos(x[2])) + 0.0016937722479819658 * (x[6] * cos(x[2]) * cos(x[3])) - 0.0020844318761020636 * (x[6] * cos(x[3])) + 0.073423840354042841 * (x[6] * pow(cos(x[3]), 2)) - 0.10255869764407463 * (pow(x[6], 2) * sin(x[2])) - 0.00014675470344092915 * (x[7] * sin(x[2]) * sin(x[3])) + 2.4143350725201305e-05 * (x[7] * pow(sin(x[2]), 2)) + 1.0891086859192866e-05 * (x[7] * pow(sin(x[3]), 2)) - 0.00018459355870316878 * (x[7] * cos(x[2])) - 0.0001596672685213968 * (x[7] * cos(x[2]) * cos(x[3])) + 1.0654482241917294e-05 * (x[7] * cos(x[3])) - 3.1258275312225509e-05 * (x[7] * pow(cos(x[3]), 2)) - 0.078919289002800944 * (pow(x[7], 2) * sin(x[2])) - 0.4321707452378572 * (sin(x[2]) * pow(sin(x[3]), 2)) + 1.5564019369976865 * (sin(x[2]) * cos(x[2])) - 0.73547405826909218 * (sin(x[2]) * cos(x[2]) * cos(x[3])) - 1.5352900739413957 * (sin(x[2]) * pow(cos(x[2]), 2)) + 42.984364896010824 * (sin(x[2]) * cos(x[3])) + 0.019709904259417082 * (sin(x[2]) * pow(cos(x[3]), 2)) - 0.25008791518677898 * pow(sin(x[2]), 3) + 72.27385752280091 * sin(x[2])) * ((sin(x[2]) * sin(x[3])) / 0.90000000000000002)) + 10 * (( - 0.0008566295437115515 * x[1] - 2.1919220402320487e-05 * x[4] - 0.0063134920963305342 * x[5] + 0.00095370464881419312 * x[6] + 0.051636005134718201 * x[7] + 6.9984537829782217e-06 * (x[0] * x[1] * x[4]) - 8.6093074215917243e-05 * (x[0] * x[4] * sin(x[3])) + 6.1823058289061733e-06 * (x[0] * x[6] * sin(x[3])) + 4.9904109234902733e-06 * (x[0] * x[7] * sin(x[2])) - 2.0250349642322381e-06 * (x[0] * sin(x[2]) * sin(x[3])) - 8.091012050541711e-06 * (pow(x[0], 2) * x[1]) - 0.00011581776240007176 * (pow(x[0], 2) * x[5]) - 5.8186288618695527e-05 * (pow(x[0], 2) * x[7]) + 0.0850852228825731 * (pow(x[0], 2) * sin(x[3])) - 9.0435792457406537e-05 * (x[1] * x[5] * sin(x[3])) - 4.0175867984229744e-06 * (x[1] * x[6] * sin(x[2])) + 5.4696454955853613e-06 * (x[1] * x[7] * sin(x[3])) + 0.00043619929115585866 * (x[1] * pow(sin(x[2]), 2)) + 3.5458812906023967e-05 * (x[1] * pow(sin(x[3]), 2)) + 6.1044524942838296e-05 * (x[1] * cos(x[2])) + 0.0020146727468697387 * (x[1] * cos(x[2]) * cos(x[3])) + 0.00078834067029993091 * (x[1] * pow(cos(x[2]), 2)) - 1.8187190306688987e-05 * (x[1] * cos(x[3])) + 3.2098954311605803e-05 * (x[1] * pow(cos(x[3]), 2)) - 0.00011164772388958151 * (pow(x[1], 2) * x[5]) - 1.3321451989029683e-05 * (pow(x[1], 2) * x[7]) + 0.092148225584680415 * (pow(x[1], 2) * sin(x[3])) + 1.0269024881732043e-05 * (x[4] * x[6] * sin(x[3])) + 2.6294903227282986e-06 * (x[4] * x[7] * sin(x[2])) - 2.4593108899959271e-05 * (x[4] * sin(x[2]) * sin(x[3])) + 1.2343239101463873e-05 * (x[4] * pow(sin(x[2]), 2)) + 1.6336381033981899e-06 * (x[4] * pow(sin(x[3]), 2)) - 1.124897285814006e-05 * (x[4] * cos(x[2])) - 1.9368052393821935e-05 * (x[4] * cos(x[2]) * cos(x[3])) - 2.0862469667145969e-05 * (x[4] * pow(cos(x[2]), 2)) - 1.5166495839347087e-06 * (x[4] * cos(x[3])) - 1.0615193579000283e-06 * (x[4] * pow(cos(x[3]), 2)) - 1.9957628207751896e-06 * (pow(x[4], 2) * x[5]) + 7.5025681097496959e-05 * (pow(x[4], 2) * x[7]) + 0.071807880144411959 * (pow(x[4], 2) * sin(x[3])) + 6.0701593606344377e-06 * (x[5] * x[6] * sin(x[2])) - 2.3397115294004764e-05 * (x[5] * pow(x[7], 2)) + 0.011803064562443325 * (x[5] * pow(sin(x[2]), 2)) + 0.00025590203268803128 * (x[5] * pow(sin(x[3]), 2)) - 0.0012586343227540559 * (x[5] * cos(x[2])) + 0.24432941235194164 * (x[5] * cos(x[2]) * cos(x[3])) + 0.0094161749899277763 * (x[5] * pow(cos(x[2]), 2)) - 0.00040709653618897779 * (x[5] * cos(x[3])) + 0.00089295388605508286 * (x[5] * pow(cos(x[3]), 2)) + 7.2100335269445267e-05 * (pow(x[5], 2) * x[7]) + 0.053246493510151295 * (pow(x[5], 2) * sin(x[3])) - 0.15783857800560189 * (x[6] * x[7] * sin(x[2])) - 0.00014675470344092915 * (x[6] * sin(x[2]) * sin(x[3])) + 2.4143350725201305e-05 * (x[6] * pow(sin(x[2]), 2)) + 1.0891086859192866e-05 * (x[6] * pow(sin(x[3]), 2)) - 0.00018459355870316878 * (x[6] * cos(x[2])) - 0.0001596672685213968 * (x[6] * cos(x[2]) * cos(x[3])) + 1.0654482241917294e-05 * (x[6] * cos(x[3])) - 3.1258275312225509e-05 * (x[6] * pow(cos(x[3]), 2)) + 0.00018318601467327039 * (pow(x[6], 2) * sin(x[3])) + 4.9213346571110482e-06 * (x[7] * sin(x[2]) * sin(x[3])) - 0.077178246641824294 * (x[7] * pow(sin(x[2]), 2)) - 0.011331055271296638 * (x[7] * pow(sin(x[3]), 2)) - 0.0018545258435467547 * (x[7] * cos(x[2])) - 0.0056441940953833912 * (x[7] * cos(x[2]) * cos(x[3])) + 9.796805138201508 * (x[7] * pow(cos(x[2]), 2)) + 0.00027279653377800206 * (x[7] * cos(x[3])) + 0.00012609536887976517 * (x[7] * pow(cos(x[3]), 2)) + 0.018596560971863487 * (pow(x[7], 2) * sin(x[3])) + 0.083029537183890084 * (pow(sin(x[2]), 2) * sin(x[3])) + 43.00959183240299 * (sin(x[3]) * cos(x[2])) + 0.91817295262933674 * (sin(x[3]) * cos(x[2]) * cos(x[3])) + 68.296246951097729 * (sin(x[3]) * pow(cos(x[2]), 2)) + 0.0082111028105765343 * (sin(x[3]) * cos(x[3])) + 0.21830873966515735 * (sin(x[3]) * pow(cos(x[3]), 2)) - 7.1311711586800999e-06 * pow(x[1], 3) - 1.1491345926482223e-06 * pow(x[5], 3) + 5.290522284224498e-06 * pow(x[7], 3) + 0.40081340811141203 * pow(sin(x[3]), 3) + 1.0462160804427931 * sin(x[3])) * ((-1 * cos(x[3])) / (0.90000000000000002 * cos(x[2])))))),
        #                  ( - (pow(x[7], 2) * sin(x[2]) * cos(x[2])) - (((-0.5 * (10 * ( - 0.0070255632221867936 * x[0] + 0.43274946929264047 * x[4] + 3.5370873498777211e-05 * x[5] + 1.0684296952698988 * x[6] - 2.1919220402320487e-05 * x[7] - 0.00072184630696900355 * (x[0] * x[1] * x[5]) + 6.9984537829782217e-06 * (x[0] * x[1] * x[7]) - 3.4591596969853197e-06 * (x[0] * x[1] * sin(x[3])) - 2.4434572886930464e-05 * (x[0] * pow(x[1], 2)) - 1.5935558915818625e-05 * (x[0] * x[4] * sin(x[2])) + 0.00027906707599179015 * (x[0] * pow(x[4], 2)) - 7.8968118646176109e-06 * (x[0] * x[5] * sin(x[3])) + 0.00018389708979083498 * (x[0] * pow(x[5], 2)) - 0.00073781288073681195 * (x[0] * x[6] * sin(x[2])) - 2.925407469212206e-05 * (x[0] * x[6] * sin(x[3])) - 8.6093074215917243e-05 * (x[0] * x[7] * sin(x[3])) + 0.12218739661935774 * (x[0] * pow(sin(x[2]), 2)) - 0.013393504249813403 * (x[0] * pow(sin(x[3]), 2)) - 0.091939144189822478 * (x[0] * cos(x[2])) + 0.00014211435304352873 * (x[0] * cos(x[2]) * cos(x[3])) + 0.094449350252944445 * (x[0] * pow(cos(x[2]), 2)) + 0.016720496219958907 * (x[0] * cos(x[3])) - 0.014351714648301811 * (x[0] * pow(cos(x[3]), 2)) - 0.0086646908040072558 * (pow(x[0], 2) * x[4]) - 3.3614392503734857e-05 * (pow(x[0], 2) * sin(x[2])) + 0.00037198209598138699 * (x[1] * x[4] * x[5]) - 8.6879398306825367e-06 * (x[1] * x[4] * sin(x[3])) - 4.6259162491407557e-06 * (x[1] * x[5] * sin(x[2])) + 3.0971893886936787e-06 * (x[1] * x[6] * sin(x[3])) - 4.1276919764238341e-06 * (x[1] * sin(x[2]) * sin(x[3])) + 0.0023129998173785997 * (pow(x[1], 2) * x[4]) - 3.5114424889291614e-05 * (pow(x[1], 2) * sin(x[2])) - 3.9915256415503791e-06 * (x[4] * x[5] * x[7]) + 5.4130420663419832e-06 * (x[4] * x[5] * sin(x[3])) + 0.091224557138273626 * (x[4] * pow(x[5], 2)) - 0.07468123277600737 * (x[4] * x[6] * sin(x[2])) + 0.14361576028882392 * (x[4] * x[7] * sin(x[3])) + 7.5025681097496959e-05 * (x[4] * pow(x[7], 2)) + 0.0034898095519765031 * (x[4] * sin(x[2]) * sin(x[3])) + 0.066569319400862345 * (x[4] * pow(sin(x[2]), 2)) + 0.36528825942112469 * (x[4] * pow(sin(x[3]), 2)) - 0.086333871747806323 * (x[4] * cos(x[2])) - 0.043069557043361448 * (x[4] * cos(x[2]) * cos(x[3])) + 0.79505652787013004 * (x[4] * pow(cos(x[2]), 2)) + 0.0033188004827935914 * (x[4] * cos(x[3])) + 0.21278195531873381 * (x[4] * pow(cos(x[3]), 2)) - 0.00018056868724906764 * (pow(x[4], 2) * sin(x[2])) + 1.3229141588693226e-06 * (x[5] * x[6] * sin(x[3])) - 0.00046638731657011118 * (x[5] * sin(x[2]) * sin(x[3])) - 3.5605819390869218e-06 * (x[5] * pow(sin(x[2]), 2)) - 2.2632939944443235e-05 * (x[5] * pow(sin(x[3]), 2)) - 4.1751174868883975e-05 * (x[5] * cos(x[2]) * cos(x[3])) - 3.0836430809464263e-06 * (x[5] * pow(cos(x[2]), 2)) + 1.5448073387723312e-05 * (x[5] * cos(x[3])) - 2.3479288291093966e-05 * (x[5] * pow(cos(x[3]), 2)) - 7.4732294079478368e-05 * (pow(x[5], 2) * sin(x[2])) + 1.0269024881732043e-05 * (x[6] * x[7] * sin(x[3])) + 1.4715341367685285e-05 * (x[6] * sin(x[2]) * sin(x[3])) + 8.9228799027727701e-06 * (x[6] * pow(sin(x[2]), 2)) + 0.0014043403211331732 * (x[6] * pow(sin(x[3]), 2)) + 0.013280304365897614 * (x[6] * cos(x[2])) + 0.3083028917750289 * (x[6] * cos(x[2]) * cos(x[3])) + 0.024251762935323457 * (x[6] * cos(x[3])) + 0.0022663726838885426 * (x[6] * pow(cos(x[3]), 2)) - 1.4365282530359833e-05 * (pow(x[6], 2) * sin(x[2])) - 2.4593108899959271e-05 * (x[7] * sin(x[2]) * sin(x[3])) + 1.2343239101463873e-05 * (x[7] * pow(sin(x[2]), 2)) + 1.6336381033981899e-06 * (x[7] * pow(sin(x[3]), 2)) - 1.124897285814006e-05 * (x[7] * cos(x[2])) - 1.9368052393821935e-05 * (x[7] * cos(x[2]) * cos(x[3])) - 2.0862469667145969e-05 * (x[7] * pow(cos(x[2]), 2)) - 1.5166495839347087e-06 * (x[7] * cos(x[3])) - 1.0615193579000283e-06 * (x[7] * pow(cos(x[3]), 2)) + 1.3147451613641493e-06 * (pow(x[7], 2) * sin(x[2])) - 0.0032281880387797545 * (sin(x[2]) * pow(sin(x[3]), 2)) + 0.00029121976419156144 * (sin(x[2]) * cos(x[2])) + 0.00013129780089208464 * (sin(x[2]) * cos(x[2]) * cos(x[3])) - 0.00029945293048839322 * (sin(x[2]) * pow(cos(x[2]), 2)) + 0.00054758808043536844 * (sin(x[2]) * cos(x[3])) - 0.0028217641589934508 * (sin(x[2]) * pow(cos(x[3]), 2)) + 0.0058769542645504928 * pow(x[0], 3) + 0.29987121206049672 * pow(x[4], 3) - 0.00034017094657775996 * pow(sin(x[2]), 3) - 0.002076395707890478 * sin(x[2])) + 10 * ((0.00038532003574531912 * x[0] + 1.0684296952698988 * x[4] + 3.2580943371291799e-05 * x[5] + 14.806257609508169 * x[6] + 0.00095370464881419312 * x[7] - 0.00073781288073681195 * (x[0] * x[4] * sin(x[2])) - 2.925407469212206e-05 * (x[0] * x[4] * sin(x[3])) + 3.2547034230164453e-06 * (x[0] * x[5] * sin(x[3])) + 5.924771687231811e-05 * (x[0] * x[6] * sin(x[2])) + 6.1823058289061733e-06 * (x[0] * x[7] * sin(x[3])) - 2.1319484037325249e-06 * (x[0] * pow(sin(x[2]), 2)) + 7.4652077840916482e-05 * (x[0] * pow(sin(x[3]), 2)) + 0.00032039783441772897 * (x[0] * cos(x[2])) + 0.00051684766904027204 * (x[0] * cos(x[2]) * cos(x[3])) + 0.00065050390476681819 * (x[0] * cos(x[3])) + 0.00016017971066541138 * (x[0] * pow(cos(x[3]), 2)) - 0.089841641284317611 * (pow(x[0], 2) * sin(x[2])) + 3.0971893886936787e-06 * (x[1] * x[4] * sin(x[3])) - 0.00072088101472899541 * (x[1] * x[5] * sin(x[2])) - 2.9055569629645779e-05 * (x[1] * x[5] * sin(x[3])) - 0.00010713489047805953 * (x[1] * x[6] * sin(x[3])) - 4.0175867984229744e-06 * (x[1] * x[7] * sin(x[2])) - 0.00051872812773371866 * (x[1] * sin(x[2]) * sin(x[3])) - 0.094007191018092975 * (pow(x[1], 2) * sin(x[2])) + 1.3229141588693226e-06 * (x[4] * x[5] * sin(x[3])) - 2.8730565060719665e-05 * (x[4] * x[6] * sin(x[2])) + 1.0269024881732043e-05 * (x[4] * x[7] * sin(x[3])) + 1.4715341367685285e-05 * (x[4] * sin(x[2]) * sin(x[3])) + 8.9228799027727701e-06 * (x[4] * pow(sin(x[2]), 2)) + 0.0014043403211331732 * (x[4] * pow(sin(x[3]), 2)) + 0.013280304365897614 * (x[4] * cos(x[2])) + 0.3083028917750289 * (x[4] * cos(x[2]) * cos(x[3])) + 0.024251762935323457 * (x[4] * cos(x[3])) + 0.0022663726838885426 * (x[4] * pow(cos(x[3]), 2)) - 0.037340616388003685 * (pow(x[4], 2) * sin(x[2])) + 0.00016693115962133971 * (x[5] * x[6] * sin(x[3])) + 6.0701593606344377e-06 * (x[5] * x[7] * sin(x[2])) - 0.0049833155971441669 * (x[5] * sin(x[2]) * sin(x[3])) - 6.3636166956642285e-06 * (x[5] * pow(sin(x[3]), 2)) - 3.772251843062457e-05 * (x[5] * cos(x[2]) * cos(x[3])) - 3.0261503939696959e-05 * (x[5] * cos(x[3])) - 1.0442392526008848e-05 * (x[5] * pow(cos(x[3]), 2)) - 0.03737958817653677 * (pow(x[5], 2) * sin(x[2])) + 0.00036637202934654078 * (x[6] * x[7] * sin(x[3])) - 0.0012271788478589803 * (x[6] * sin(x[2]) * sin(x[3])) - 0.00036033240030784372 * (x[6] * pow(sin(x[2]), 2)) + 0.059746490847326778 * (x[6] * pow(sin(x[3]), 2)) - 7.5040755460096105e-05 * (x[6] * cos(x[2])) + 0.0016937722479819658 * (x[6] * cos(x[2]) * cos(x[3])) - 0.0020844318761020636 * (x[6] * cos(x[3])) + 0.073423840354042841 * (x[6] * pow(cos(x[3]), 2)) - 0.10255869764407463 * (pow(x[6], 2) * sin(x[2])) - 0.00014675470344092915 * (x[7] * sin(x[2]) * sin(x[3])) + 2.4143350725201305e-05 * (x[7] * pow(sin(x[2]), 2)) + 1.0891086859192866e-05 * (x[7] * pow(sin(x[3]), 2)) - 0.00018459355870316878 * (x[7] * cos(x[2])) - 0.0001596672685213968 * (x[7] * cos(x[2]) * cos(x[3])) + 1.0654482241917294e-05 * (x[7] * cos(x[3])) - 3.1258275312225509e-05 * (x[7] * pow(cos(x[3]), 2)) - 0.078919289002800944 * (pow(x[7], 2) * sin(x[2])) - 0.4321707452378572 * (sin(x[2]) * pow(sin(x[3]), 2)) + 1.5564019369976865 * (sin(x[2]) * cos(x[2])) - 0.73547405826909218 * (sin(x[2]) * cos(x[2]) * cos(x[3])) - 1.5352900739413957 * (sin(x[2]) * pow(cos(x[2]), 2)) + 42.984364896010824 * (sin(x[2]) * cos(x[3])) + 0.019709904259417082 * (sin(x[2]) * pow(cos(x[3]), 2)) - 0.25008791518677898 * pow(sin(x[2]), 3) + 72.27385752280091 * sin(x[2])) * ((-1 * cos(x[3])) / 0.90000000000000002)))) / 0.90000000000000002) * cos(x[2])) + (((-0.5 * (10 * ( - 0.0065597048649271477 * x[1] + 3.5370873498777211e-05 * x[4] + 0.18283061060819256 * x[5] + 3.2580943371291799e-05 * x[6] - 0.0063134920963305342 * x[7] - 0.00072184630696900355 * (x[0] * x[1] * x[4]) + 3.2449042204134365e-06 * (x[0] * x[1] * sin(x[2])) + 0.00036779417958166996 * (x[0] * x[4] * x[5]) - 7.8968118646176109e-06 * (x[0] * x[4] * sin(x[3])) + 5.6273598214134143e-06 * (x[0] * x[5] * sin(x[2])) + 3.2547034230164453e-06 * (x[0] * x[6] * sin(x[3])) - 1.0186598971223059e-05 * (x[0] * sin(x[2]) * sin(x[3])) - 2.8594868097985197e-05 * (pow(x[0], 2) * x[1]) + 0.0012409276895305038 * (pow(x[0], 2) * x[5]) - 0.00011581776240007176 * (pow(x[0], 2) * x[7]) + 3.7192594459799988e-05 * (pow(x[0], 2) * sin(x[3])) - 4.6259162491407557e-06 * (x[1] * x[4] * sin(x[2])) + 0.00018599104799069349 * (x[1] * pow(x[4], 2)) - 2.0707620350439519e-05 * (x[1] * x[5] * sin(x[3])) + 0.00029670443804646895 * (x[1] * pow(x[5], 2)) - 0.00072088101472899541 * (x[1] * x[6] * sin(x[2])) - 2.9055569629645779e-05 * (x[1] * x[6] * sin(x[3])) - 9.0435792457406537e-05 * (x[1] * x[7] * sin(x[3])) + 0.12156527713545084 * (x[1] * pow(sin(x[2]), 2)) - 0.011546315046333411 * (x[1] * pow(sin(x[3]), 2)) - 0.087966473726447128 * (x[1] * cos(x[2])) + 0.0014347684052675109 * (x[1] * cos(x[2]) * cos(x[3])) + 0.090639885085965918 * (x[1] * pow(cos(x[2]), 2)) + 0.015068584266482114 * (x[1] * cos(x[3])) - 0.01245848954964501 * (x[1] * pow(cos(x[3]), 2)) - 0.0090866721254703152 * (pow(x[1], 2) * x[5]) - 0.00011164772388958151 * (pow(x[1], 2) * x[7]) + 3.3894269510030535e-05 * (pow(x[1], 2) * sin(x[3])) - 0.00014946458815895674 * (x[4] * x[5] * sin(x[2])) + 1.3229141588693226e-06 * (x[4] * x[6] * sin(x[3])) - 0.00046638731657011118 * (x[4] * sin(x[2]) * sin(x[3])) - 3.5605819390869218e-06 * (x[4] * pow(sin(x[2]), 2)) - 2.2632939944443235e-05 * (x[4] * pow(sin(x[3]), 2)) - 4.1751174868883975e-05 * (x[4] * cos(x[2]) * cos(x[3])) - 3.0836430809464263e-06 * (x[4] * pow(cos(x[2]), 2)) + 1.5448073387723312e-05 * (x[4] * cos(x[3])) - 2.3479288291093966e-05 * (x[4] * pow(cos(x[3]), 2)) + 0.091224557138273626 * (pow(x[4], 2) * x[5]) - 1.9957628207751896e-06 * (pow(x[4], 2) * x[7]) + 2.7065210331709916e-06 * (pow(x[4], 2) * sin(x[3])) - 0.074759176353073539 * (x[5] * x[6] * sin(x[2])) + 0.10649298702030259 * (x[5] * x[7] * sin(x[3])) + 7.2100335269445267e-05 * (x[5] * pow(x[7], 2)) + 0.0036976152822558923 * (x[5] * sin(x[2]) * sin(x[3])) + 0.074265982189716306 * (x[5] * pow(sin(x[2]), 2)) + 0.45834448690085838 * (x[5] * pow(sin(x[3]), 2)) - 0.14778765323579626 * (x[5] * cos(x[2])) - 0.045042136635330805 * (x[5] * cos(x[2]) * cos(x[3])) + 0.222497791565338 * (x[5] * pow(cos(x[2]), 2)) - 0.0030011200314892816 * (x[5] * cos(x[3])) + 0.31764079677561585 * (x[5] * pow(cos(x[3]), 2)) - 3.4474037779446669e-06 * (pow(x[5], 2) * x[7]) + 1.2142390048262233e-05 * (pow(x[5], 2) * sin(x[3])) + 6.0701593606344377e-06 * (x[6] * x[7] * sin(x[2])) - 0.0049833155971441669 * (x[6] * sin(x[2]) * sin(x[3])) - 6.3636166956642285e-06 * (x[6] * pow(sin(x[3]), 2)) - 3.772251843062457e-05 * (x[6] * cos(x[2]) * cos(x[3])) - 3.0261503939696959e-05 * (x[6] * cos(x[3])) - 1.0442392526008848e-05 * (x[6] * pow(cos(x[3]), 2)) + 8.3465579810669855e-05 * (pow(x[6], 2) * sin(x[3])) + 0.011803064562443325 * (x[7] * pow(sin(x[2]), 2)) + 0.00025590203268803128 * (x[7] * pow(sin(x[3]), 2)) - 0.0012586343227540559 * (x[7] * cos(x[2])) + 0.24432941235194164 * (x[7] * cos(x[2]) * cos(x[3])) + 0.0094161749899277763 * (x[7] * pow(cos(x[2]), 2)) - 0.00040709653618897779 * (x[7] * cos(x[3])) + 0.00089295388605508286 * (x[7] * pow(cos(x[3]), 2)) - 0.0030776700047295234 * (pow(sin(x[2]), 2) * sin(x[3])) - 0.0056215135530187773 * (sin(x[3]) * cos(x[2])) + 0.023072964768245035 * (sin(x[3]) * cos(x[2]) * cos(x[3])) - 0.0025689842009279191 * (sin(x[3]) * pow(cos(x[2]), 2)) - 0.0023865963238517305 * (sin(x[3]) * cos(x[3])) + 0.010998779258773987 * (sin(x[3]) * pow(cos(x[3]), 2)) + 0.0053441513135730721 * pow(x[1], 3) + 0.32831137118854503 * pow(x[5], 3) - 7.7990384313349214e-06 * pow(x[7], 3) + 0.0071561515685901571 * pow(sin(x[3]), 3) - 0.029147178994592283 * sin(x[3])) + 10 * ((0.00038532003574531912 * x[0] + 1.0684296952698988 * x[4] + 3.2580943371291799e-05 * x[5] + 14.806257609508169 * x[6] + 0.00095370464881419312 * x[7] - 0.00073781288073681195 * (x[0] * x[4] * sin(x[2])) - 2.925407469212206e-05 * (x[0] * x[4] * sin(x[3])) + 3.2547034230164453e-06 * (x[0] * x[5] * sin(x[3])) + 5.924771687231811e-05 * (x[0] * x[6] * sin(x[2])) + 6.1823058289061733e-06 * (x[0] * x[7] * sin(x[3])) - 2.1319484037325249e-06 * (x[0] * pow(sin(x[2]), 2)) + 7.4652077840916482e-05 * (x[0] * pow(sin(x[3]), 2)) + 0.00032039783441772897 * (x[0] * cos(x[2])) + 0.00051684766904027204 * (x[0] * cos(x[2]) * cos(x[3])) + 0.00065050390476681819 * (x[0] * cos(x[3])) + 0.00016017971066541138 * (x[0] * pow(cos(x[3]), 2)) - 0.089841641284317611 * (pow(x[0], 2) * sin(x[2])) + 3.0971893886936787e-06 * (x[1] * x[4] * sin(x[3])) - 0.00072088101472899541 * (x[1] * x[5] * sin(x[2])) - 2.9055569629645779e-05 * (x[1] * x[5] * sin(x[3])) - 0.00010713489047805953 * (x[1] * x[6] * sin(x[3])) - 4.0175867984229744e-06 * (x[1] * x[7] * sin(x[2])) - 0.00051872812773371866 * (x[1] * sin(x[2]) * sin(x[3])) - 0.094007191018092975 * (pow(x[1], 2) * sin(x[2])) + 1.3229141588693226e-06 * (x[4] * x[5] * sin(x[3])) - 2.8730565060719665e-05 * (x[4] * x[6] * sin(x[2])) + 1.0269024881732043e-05 * (x[4] * x[7] * sin(x[3])) + 1.4715341367685285e-05 * (x[4] * sin(x[2]) * sin(x[3])) + 8.9228799027727701e-06 * (x[4] * pow(sin(x[2]), 2)) + 0.0014043403211331732 * (x[4] * pow(sin(x[3]), 2)) + 0.013280304365897614 * (x[4] * cos(x[2])) + 0.3083028917750289 * (x[4] * cos(x[2]) * cos(x[3])) + 0.024251762935323457 * (x[4] * cos(x[3])) + 0.0022663726838885426 * (x[4] * pow(cos(x[3]), 2)) - 0.037340616388003685 * (pow(x[4], 2) * sin(x[2])) + 0.00016693115962133971 * (x[5] * x[6] * sin(x[3])) + 6.0701593606344377e-06 * (x[5] * x[7] * sin(x[2])) - 0.0049833155971441669 * (x[5] * sin(x[2]) * sin(x[3])) - 6.3636166956642285e-06 * (x[5] * pow(sin(x[3]), 2)) - 3.772251843062457e-05 * (x[5] * cos(x[2]) * cos(x[3])) - 3.0261503939696959e-05 * (x[5] * cos(x[3])) - 1.0442392526008848e-05 * (x[5] * pow(cos(x[3]), 2)) - 0.03737958817653677 * (pow(x[5], 2) * sin(x[2])) + 0.00036637202934654078 * (x[6] * x[7] * sin(x[3])) - 0.0012271788478589803 * (x[6] * sin(x[2]) * sin(x[3])) - 0.00036033240030784372 * (x[6] * pow(sin(x[2]), 2)) + 0.059746490847326778 * (x[6] * pow(sin(x[3]), 2)) - 7.5040755460096105e-05 * (x[6] * cos(x[2])) + 0.0016937722479819658 * (x[6] * cos(x[2]) * cos(x[3])) - 0.0020844318761020636 * (x[6] * cos(x[3])) + 0.073423840354042841 * (x[6] * pow(cos(x[3]), 2)) - 0.10255869764407463 * (pow(x[6], 2) * sin(x[2])) - 0.00014675470344092915 * (x[7] * sin(x[2]) * sin(x[3])) + 2.4143350725201305e-05 * (x[7] * pow(sin(x[2]), 2)) + 1.0891086859192866e-05 * (x[7] * pow(sin(x[3]), 2)) - 0.00018459355870316878 * (x[7] * cos(x[2])) - 0.0001596672685213968 * (x[7] * cos(x[2]) * cos(x[3])) + 1.0654482241917294e-05 * (x[7] * cos(x[3])) - 3.1258275312225509e-05 * (x[7] * pow(cos(x[3]), 2)) - 0.078919289002800944 * (pow(x[7], 2) * sin(x[2])) - 0.4321707452378572 * (sin(x[2]) * pow(sin(x[3]), 2)) + 1.5564019369976865 * (sin(x[2]) * cos(x[2])) - 0.73547405826909218 * (sin(x[2]) * cos(x[2]) * cos(x[3])) - 1.5352900739413957 * (sin(x[2]) * pow(cos(x[2]), 2)) + 42.984364896010824 * (sin(x[2]) * cos(x[3])) + 0.019709904259417082 * (sin(x[2]) * pow(cos(x[3]), 2)) - 0.25008791518677898 * pow(sin(x[2]), 3) + 72.27385752280091 * sin(x[2])) * ((sin(x[2]) * sin(x[3])) / 0.90000000000000002)) + 10 * (( - 0.0008566295437115515 * x[1] - 2.1919220402320487e-05 * x[4] - 0.0063134920963305342 * x[5] + 0.00095370464881419312 * x[6] + 0.051636005134718201 * x[7] + 6.9984537829782217e-06 * (x[0] * x[1] * x[4]) - 8.6093074215917243e-05 * (x[0] * x[4] * sin(x[3])) + 6.1823058289061733e-06 * (x[0] * x[6] * sin(x[3])) + 4.9904109234902733e-06 * (x[0] * x[7] * sin(x[2])) - 2.0250349642322381e-06 * (x[0] * sin(x[2]) * sin(x[3])) - 8.091012050541711e-06 * (pow(x[0], 2) * x[1]) - 0.00011581776240007176 * (pow(x[0], 2) * x[5]) - 5.8186288618695527e-05 * (pow(x[0], 2) * x[7]) + 0.0850852228825731 * (pow(x[0], 2) * sin(x[3])) - 9.0435792457406537e-05 * (x[1] * x[5] * sin(x[3])) - 4.0175867984229744e-06 * (x[1] * x[6] * sin(x[2])) + 5.4696454955853613e-06 * (x[1] * x[7] * sin(x[3])) + 0.00043619929115585866 * (x[1] * pow(sin(x[2]), 2)) + 3.5458812906023967e-05 * (x[1] * pow(sin(x[3]), 2)) + 6.1044524942838296e-05 * (x[1] * cos(x[2])) + 0.0020146727468697387 * (x[1] * cos(x[2]) * cos(x[3])) + 0.00078834067029993091 * (x[1] * pow(cos(x[2]), 2)) - 1.8187190306688987e-05 * (x[1] * cos(x[3])) + 3.2098954311605803e-05 * (x[1] * pow(cos(x[3]), 2)) - 0.00011164772388958151 * (pow(x[1], 2) * x[5]) - 1.3321451989029683e-05 * (pow(x[1], 2) * x[7]) + 0.092148225584680415 * (pow(x[1], 2) * sin(x[3])) + 1.0269024881732043e-05 * (x[4] * x[6] * sin(x[3])) + 2.6294903227282986e-06 * (x[4] * x[7] * sin(x[2])) - 2.4593108899959271e-05 * (x[4] * sin(x[2]) * sin(x[3])) + 1.2343239101463873e-05 * (x[4] * pow(sin(x[2]), 2)) + 1.6336381033981899e-06 * (x[4] * pow(sin(x[3]), 2)) - 1.124897285814006e-05 * (x[4] * cos(x[2])) - 1.9368052393821935e-05 * (x[4] * cos(x[2]) * cos(x[3])) - 2.0862469667145969e-05 * (x[4] * pow(cos(x[2]), 2)) - 1.5166495839347087e-06 * (x[4] * cos(x[3])) - 1.0615193579000283e-06 * (x[4] * pow(cos(x[3]), 2)) - 1.9957628207751896e-06 * (pow(x[4], 2) * x[5]) + 7.5025681097496959e-05 * (pow(x[4], 2) * x[7]) + 0.071807880144411959 * (pow(x[4], 2) * sin(x[3])) + 6.0701593606344377e-06 * (x[5] * x[6] * sin(x[2])) - 2.3397115294004764e-05 * (x[5] * pow(x[7], 2)) + 0.011803064562443325 * (x[5] * pow(sin(x[2]), 2)) + 0.00025590203268803128 * (x[5] * pow(sin(x[3]), 2)) - 0.0012586343227540559 * (x[5] * cos(x[2])) + 0.24432941235194164 * (x[5] * cos(x[2]) * cos(x[3])) + 0.0094161749899277763 * (x[5] * pow(cos(x[2]), 2)) - 0.00040709653618897779 * (x[5] * cos(x[3])) + 0.00089295388605508286 * (x[5] * pow(cos(x[3]), 2)) + 7.2100335269445267e-05 * (pow(x[5], 2) * x[7]) + 0.053246493510151295 * (pow(x[5], 2) * sin(x[3])) - 0.15783857800560189 * (x[6] * x[7] * sin(x[2])) - 0.00014675470344092915 * (x[6] * sin(x[2]) * sin(x[3])) + 2.4143350725201305e-05 * (x[6] * pow(sin(x[2]), 2)) + 1.0891086859192866e-05 * (x[6] * pow(sin(x[3]), 2)) - 0.00018459355870316878 * (x[6] * cos(x[2])) - 0.0001596672685213968 * (x[6] * cos(x[2]) * cos(x[3])) + 1.0654482241917294e-05 * (x[6] * cos(x[3])) - 3.1258275312225509e-05 * (x[6] * pow(cos(x[3]), 2)) + 0.00018318601467327039 * (pow(x[6], 2) * sin(x[3])) + 4.9213346571110482e-06 * (x[7] * sin(x[2]) * sin(x[3])) - 0.077178246641824294 * (x[7] * pow(sin(x[2]), 2)) - 0.011331055271296638 * (x[7] * pow(sin(x[3]), 2)) - 0.0018545258435467547 * (x[7] * cos(x[2])) - 0.0056441940953833912 * (x[7] * cos(x[2]) * cos(x[3])) + 9.796805138201508 * (x[7] * pow(cos(x[2]), 2)) + 0.00027279653377800206 * (x[7] * cos(x[3])) + 0.00012609536887976517 * (x[7] * pow(cos(x[3]), 2)) + 0.018596560971863487 * (pow(x[7], 2) * sin(x[3])) + 0.083029537183890084 * (pow(sin(x[2]), 2) * sin(x[3])) + 43.00959183240299 * (sin(x[3]) * cos(x[2])) + 0.91817295262933674 * (sin(x[3]) * cos(x[2]) * cos(x[3])) + 68.296246951097729 * (sin(x[3]) * pow(cos(x[2]), 2)) + 0.0082111028105765343 * (sin(x[3]) * cos(x[3])) + 0.21830873966515735 * (sin(x[3]) * pow(cos(x[3]), 2)) - 7.1311711586800999e-06 * pow(x[1], 3) - 1.1491345926482223e-06 * pow(x[5], 3) + 5.290522284224498e-06 * pow(x[7], 3) + 0.40081340811141203 * pow(sin(x[3]), 3) + 1.0462160804427931 * sin(x[3])) * ((-1 * cos(x[3])) / (0.90000000000000002 * cos(x[2])))))) / 0.90000000000000002) * sin(x[2]) * sin(x[3])) + 10.9 * (sin(x[2]) * cos(x[3]))),
        #                  ((2 * (x[6] * x[7] * sin(x[2])) - (((-0.5 * (10 * ( - 0.0065597048649271477 * x[1] + 3.5370873498777211e-05 * x[4] + 0.18283061060819256 * x[5] + 3.2580943371291799e-05 * x[6] - 0.0063134920963305342 * x[7] - 0.00072184630696900355 * (x[0] * x[1] * x[4]) + 3.2449042204134365e-06 * (x[0] * x[1] * sin(x[2])) + 0.00036779417958166996 * (x[0] * x[4] * x[5]) - 7.8968118646176109e-06 * (x[0] * x[4] * sin(x[3])) + 5.6273598214134143e-06 * (x[0] * x[5] * sin(x[2])) + 3.2547034230164453e-06 * (x[0] * x[6] * sin(x[3])) - 1.0186598971223059e-05 * (x[0] * sin(x[2]) * sin(x[3])) - 2.8594868097985197e-05 * (pow(x[0], 2) * x[1]) + 0.0012409276895305038 * (pow(x[0], 2) * x[5]) - 0.00011581776240007176 * (pow(x[0], 2) * x[7]) + 3.7192594459799988e-05 * (pow(x[0], 2) * sin(x[3])) - 4.6259162491407557e-06 * (x[1] * x[4] * sin(x[2])) + 0.00018599104799069349 * (x[1] * pow(x[4], 2)) - 2.0707620350439519e-05 * (x[1] * x[5] * sin(x[3])) + 0.00029670443804646895 * (x[1] * pow(x[5], 2)) - 0.00072088101472899541 * (x[1] * x[6] * sin(x[2])) - 2.9055569629645779e-05 * (x[1] * x[6] * sin(x[3])) - 9.0435792457406537e-05 * (x[1] * x[7] * sin(x[3])) + 0.12156527713545084 * (x[1] * pow(sin(x[2]), 2)) - 0.011546315046333411 * (x[1] * pow(sin(x[3]), 2)) - 0.087966473726447128 * (x[1] * cos(x[2])) + 0.0014347684052675109 * (x[1] * cos(x[2]) * cos(x[3])) + 0.090639885085965918 * (x[1] * pow(cos(x[2]), 2)) + 0.015068584266482114 * (x[1] * cos(x[3])) - 0.01245848954964501 * (x[1] * pow(cos(x[3]), 2)) - 0.0090866721254703152 * (pow(x[1], 2) * x[5]) - 0.00011164772388958151 * (pow(x[1], 2) * x[7]) + 3.3894269510030535e-05 * (pow(x[1], 2) * sin(x[3])) - 0.00014946458815895674 * (x[4] * x[5] * sin(x[2])) + 1.3229141588693226e-06 * (x[4] * x[6] * sin(x[3])) - 0.00046638731657011118 * (x[4] * sin(x[2]) * sin(x[3])) - 3.5605819390869218e-06 * (x[4] * pow(sin(x[2]), 2)) - 2.2632939944443235e-05 * (x[4] * pow(sin(x[3]), 2)) - 4.1751174868883975e-05 * (x[4] * cos(x[2]) * cos(x[3])) - 3.0836430809464263e-06 * (x[4] * pow(cos(x[2]), 2)) + 1.5448073387723312e-05 * (x[4] * cos(x[3])) - 2.3479288291093966e-05 * (x[4] * pow(cos(x[3]), 2)) + 0.091224557138273626 * (pow(x[4], 2) * x[5]) - 1.9957628207751896e-06 * (pow(x[4], 2) * x[7]) + 2.7065210331709916e-06 * (pow(x[4], 2) * sin(x[3])) - 0.074759176353073539 * (x[5] * x[6] * sin(x[2])) + 0.10649298702030259 * (x[5] * x[7] * sin(x[3])) + 7.2100335269445267e-05 * (x[5] * pow(x[7], 2)) + 0.0036976152822558923 * (x[5] * sin(x[2]) * sin(x[3])) + 0.074265982189716306 * (x[5] * pow(sin(x[2]), 2)) + 0.45834448690085838 * (x[5] * pow(sin(x[3]), 2)) - 0.14778765323579626 * (x[5] * cos(x[2])) - 0.045042136635330805 * (x[5] * cos(x[2]) * cos(x[3])) + 0.222497791565338 * (x[5] * pow(cos(x[2]), 2)) - 0.0030011200314892816 * (x[5] * cos(x[3])) + 0.31764079677561585 * (x[5] * pow(cos(x[3]), 2)) - 3.4474037779446669e-06 * (pow(x[5], 2) * x[7]) + 1.2142390048262233e-05 * (pow(x[5], 2) * sin(x[3])) + 6.0701593606344377e-06 * (x[6] * x[7] * sin(x[2])) - 0.0049833155971441669 * (x[6] * sin(x[2]) * sin(x[3])) - 6.3636166956642285e-06 * (x[6] * pow(sin(x[3]), 2)) - 3.772251843062457e-05 * (x[6] * cos(x[2]) * cos(x[3])) - 3.0261503939696959e-05 * (x[6] * cos(x[3])) - 1.0442392526008848e-05 * (x[6] * pow(cos(x[3]), 2)) + 8.3465579810669855e-05 * (pow(x[6], 2) * sin(x[3])) + 0.011803064562443325 * (x[7] * pow(sin(x[2]), 2)) + 0.00025590203268803128 * (x[7] * pow(sin(x[3]), 2)) - 0.0012586343227540559 * (x[7] * cos(x[2])) + 0.24432941235194164 * (x[7] * cos(x[2]) * cos(x[3])) + 0.0094161749899277763 * (x[7] * pow(cos(x[2]), 2)) - 0.00040709653618897779 * (x[7] * cos(x[3])) + 0.00089295388605508286 * (x[7] * pow(cos(x[3]), 2)) - 0.0030776700047295234 * (pow(sin(x[2]), 2) * sin(x[3])) - 0.0056215135530187773 * (sin(x[3]) * cos(x[2])) + 0.023072964768245035 * (sin(x[3]) * cos(x[2]) * cos(x[3])) - 0.0025689842009279191 * (sin(x[3]) * pow(cos(x[2]), 2)) - 0.0023865963238517305 * (sin(x[3]) * cos(x[3])) + 0.010998779258773987 * (sin(x[3]) * pow(cos(x[3]), 2)) + 0.0053441513135730721 * pow(x[1], 3) + 0.32831137118854503 * pow(x[5], 3) - 7.7990384313349214e-06 * pow(x[7], 3) + 0.0071561515685901571 * pow(sin(x[3]), 3) - 0.029147178994592283 * sin(x[3])) + 10 * ((0.00038532003574531912 * x[0] + 1.0684296952698988 * x[4] + 3.2580943371291799e-05 * x[5] + 14.806257609508169 * x[6] + 0.00095370464881419312 * x[7] - 0.00073781288073681195 * (x[0] * x[4] * sin(x[2])) - 2.925407469212206e-05 * (x[0] * x[4] * sin(x[3])) + 3.2547034230164453e-06 * (x[0] * x[5] * sin(x[3])) + 5.924771687231811e-05 * (x[0] * x[6] * sin(x[2])) + 6.1823058289061733e-06 * (x[0] * x[7] * sin(x[3])) - 2.1319484037325249e-06 * (x[0] * pow(sin(x[2]), 2)) + 7.4652077840916482e-05 * (x[0] * pow(sin(x[3]), 2)) + 0.00032039783441772897 * (x[0] * cos(x[2])) + 0.00051684766904027204 * (x[0] * cos(x[2]) * cos(x[3])) + 0.00065050390476681819 * (x[0] * cos(x[3])) + 0.00016017971066541138 * (x[0] * pow(cos(x[3]), 2)) - 0.089841641284317611 * (pow(x[0], 2) * sin(x[2])) + 3.0971893886936787e-06 * (x[1] * x[4] * sin(x[3])) - 0.00072088101472899541 * (x[1] * x[5] * sin(x[2])) - 2.9055569629645779e-05 * (x[1] * x[5] * sin(x[3])) - 0.00010713489047805953 * (x[1] * x[6] * sin(x[3])) - 4.0175867984229744e-06 * (x[1] * x[7] * sin(x[2])) - 0.00051872812773371866 * (x[1] * sin(x[2]) * sin(x[3])) - 0.094007191018092975 * (pow(x[1], 2) * sin(x[2])) + 1.3229141588693226e-06 * (x[4] * x[5] * sin(x[3])) - 2.8730565060719665e-05 * (x[4] * x[6] * sin(x[2])) + 1.0269024881732043e-05 * (x[4] * x[7] * sin(x[3])) + 1.4715341367685285e-05 * (x[4] * sin(x[2]) * sin(x[3])) + 8.9228799027727701e-06 * (x[4] * pow(sin(x[2]), 2)) + 0.0014043403211331732 * (x[4] * pow(sin(x[3]), 2)) + 0.013280304365897614 * (x[4] * cos(x[2])) + 0.3083028917750289 * (x[4] * cos(x[2]) * cos(x[3])) + 0.024251762935323457 * (x[4] * cos(x[3])) + 0.0022663726838885426 * (x[4] * pow(cos(x[3]), 2)) - 0.037340616388003685 * (pow(x[4], 2) * sin(x[2])) + 0.00016693115962133971 * (x[5] * x[6] * sin(x[3])) + 6.0701593606344377e-06 * (x[5] * x[7] * sin(x[2])) - 0.0049833155971441669 * (x[5] * sin(x[2]) * sin(x[3])) - 6.3636166956642285e-06 * (x[5] * pow(sin(x[3]), 2)) - 3.772251843062457e-05 * (x[5] * cos(x[2]) * cos(x[3])) - 3.0261503939696959e-05 * (x[5] * cos(x[3])) - 1.0442392526008848e-05 * (x[5] * pow(cos(x[3]), 2)) - 0.03737958817653677 * (pow(x[5], 2) * sin(x[2])) + 0.00036637202934654078 * (x[6] * x[7] * sin(x[3])) - 0.0012271788478589803 * (x[6] * sin(x[2]) * sin(x[3])) - 0.00036033240030784372 * (x[6] * pow(sin(x[2]), 2)) + 0.059746490847326778 * (x[6] * pow(sin(x[3]), 2)) - 7.5040755460096105e-05 * (x[6] * cos(x[2])) + 0.0016937722479819658 * (x[6] * cos(x[2]) * cos(x[3])) - 0.0020844318761020636 * (x[6] * cos(x[3])) + 0.073423840354042841 * (x[6] * pow(cos(x[3]), 2)) - 0.10255869764407463 * (pow(x[6], 2) * sin(x[2])) - 0.00014675470344092915 * (x[7] * sin(x[2]) * sin(x[3])) + 2.4143350725201305e-05 * (x[7] * pow(sin(x[2]), 2)) + 1.0891086859192866e-05 * (x[7] * pow(sin(x[3]), 2)) - 0.00018459355870316878 * (x[7] * cos(x[2])) - 0.0001596672685213968 * (x[7] * cos(x[2]) * cos(x[3])) + 1.0654482241917294e-05 * (x[7] * cos(x[3])) - 3.1258275312225509e-05 * (x[7] * pow(cos(x[3]), 2)) - 0.078919289002800944 * (pow(x[7], 2) * sin(x[2])) - 0.4321707452378572 * (sin(x[2]) * pow(sin(x[3]), 2)) + 1.5564019369976865 * (sin(x[2]) * cos(x[2])) - 0.73547405826909218 * (sin(x[2]) * cos(x[2]) * cos(x[3])) - 1.5352900739413957 * (sin(x[2]) * pow(cos(x[2]), 2)) + 42.984364896010824 * (sin(x[2]) * cos(x[3])) + 0.019709904259417082 * (sin(x[2]) * pow(cos(x[3]), 2)) - 0.25008791518677898 * pow(sin(x[2]), 3) + 72.27385752280091 * sin(x[2])) * ((sin(x[2]) * sin(x[3])) / 0.90000000000000002)) + 10 * (( - 0.0008566295437115515 * x[1] - 2.1919220402320487e-05 * x[4] - 0.0063134920963305342 * x[5] + 0.00095370464881419312 * x[6] + 0.051636005134718201 * x[7] + 6.9984537829782217e-06 * (x[0] * x[1] * x[4]) - 8.6093074215917243e-05 * (x[0] * x[4] * sin(x[3])) + 6.1823058289061733e-06 * (x[0] * x[6] * sin(x[3])) + 4.9904109234902733e-06 * (x[0] * x[7] * sin(x[2])) - 2.0250349642322381e-06 * (x[0] * sin(x[2]) * sin(x[3])) - 8.091012050541711e-06 * (pow(x[0], 2) * x[1]) - 0.00011581776240007176 * (pow(x[0], 2) * x[5]) - 5.8186288618695527e-05 * (pow(x[0], 2) * x[7]) + 0.0850852228825731 * (pow(x[0], 2) * sin(x[3])) - 9.0435792457406537e-05 * (x[1] * x[5] * sin(x[3])) - 4.0175867984229744e-06 * (x[1] * x[6] * sin(x[2])) + 5.4696454955853613e-06 * (x[1] * x[7] * sin(x[3])) + 0.00043619929115585866 * (x[1] * pow(sin(x[2]), 2)) + 3.5458812906023967e-05 * (x[1] * pow(sin(x[3]), 2)) + 6.1044524942838296e-05 * (x[1] * cos(x[2])) + 0.0020146727468697387 * (x[1] * cos(x[2]) * cos(x[3])) + 0.00078834067029993091 * (x[1] * pow(cos(x[2]), 2)) - 1.8187190306688987e-05 * (x[1] * cos(x[3])) + 3.2098954311605803e-05 * (x[1] * pow(cos(x[3]), 2)) - 0.00011164772388958151 * (pow(x[1], 2) * x[5]) - 1.3321451989029683e-05 * (pow(x[1], 2) * x[7]) + 0.092148225584680415 * (pow(x[1], 2) * sin(x[3])) + 1.0269024881732043e-05 * (x[4] * x[6] * sin(x[3])) + 2.6294903227282986e-06 * (x[4] * x[7] * sin(x[2])) - 2.4593108899959271e-05 * (x[4] * sin(x[2]) * sin(x[3])) + 1.2343239101463873e-05 * (x[4] * pow(sin(x[2]), 2)) + 1.6336381033981899e-06 * (x[4] * pow(sin(x[3]), 2)) - 1.124897285814006e-05 * (x[4] * cos(x[2])) - 1.9368052393821935e-05 * (x[4] * cos(x[2]) * cos(x[3])) - 2.0862469667145969e-05 * (x[4] * pow(cos(x[2]), 2)) - 1.5166495839347087e-06 * (x[4] * cos(x[3])) - 1.0615193579000283e-06 * (x[4] * pow(cos(x[3]), 2)) - 1.9957628207751896e-06 * (pow(x[4], 2) * x[5]) + 7.5025681097496959e-05 * (pow(x[4], 2) * x[7]) + 0.071807880144411959 * (pow(x[4], 2) * sin(x[3])) + 6.0701593606344377e-06 * (x[5] * x[6] * sin(x[2])) - 2.3397115294004764e-05 * (x[5] * pow(x[7], 2)) + 0.011803064562443325 * (x[5] * pow(sin(x[2]), 2)) + 0.00025590203268803128 * (x[5] * pow(sin(x[3]), 2)) - 0.0012586343227540559 * (x[5] * cos(x[2])) + 0.24432941235194164 * (x[5] * cos(x[2]) * cos(x[3])) + 0.0094161749899277763 * (x[5] * pow(cos(x[2]), 2)) - 0.00040709653618897779 * (x[5] * cos(x[3])) + 0.00089295388605508286 * (x[5] * pow(cos(x[3]), 2)) + 7.2100335269445267e-05 * (pow(x[5], 2) * x[7]) + 0.053246493510151295 * (pow(x[5], 2) * sin(x[3])) - 0.15783857800560189 * (x[6] * x[7] * sin(x[2])) - 0.00014675470344092915 * (x[6] * sin(x[2]) * sin(x[3])) + 2.4143350725201305e-05 * (x[6] * pow(sin(x[2]), 2)) + 1.0891086859192866e-05 * (x[6] * pow(sin(x[3]), 2)) - 0.00018459355870316878 * (x[6] * cos(x[2])) - 0.0001596672685213968 * (x[6] * cos(x[2]) * cos(x[3])) + 1.0654482241917294e-05 * (x[6] * cos(x[3])) - 3.1258275312225509e-05 * (x[6] * pow(cos(x[3]), 2)) + 0.00018318601467327039 * (pow(x[6], 2) * sin(x[3])) + 4.9213346571110482e-06 * (x[7] * sin(x[2]) * sin(x[3])) - 0.077178246641824294 * (x[7] * pow(sin(x[2]), 2)) - 0.011331055271296638 * (x[7] * pow(sin(x[3]), 2)) - 0.0018545258435467547 * (x[7] * cos(x[2])) - 0.0056441940953833912 * (x[7] * cos(x[2]) * cos(x[3])) + 9.796805138201508 * (x[7] * pow(cos(x[2]), 2)) + 0.00027279653377800206 * (x[7] * cos(x[3])) + 0.00012609536887976517 * (x[7] * pow(cos(x[3]), 2)) + 0.018596560971863487 * (pow(x[7], 2) * sin(x[3])) + 0.083029537183890084 * (pow(sin(x[2]), 2) * sin(x[3])) + 43.00959183240299 * (sin(x[3]) * cos(x[2])) + 0.91817295262933674 * (sin(x[3]) * cos(x[2]) * cos(x[3])) + 68.296246951097729 * (sin(x[3]) * pow(cos(x[2]), 2)) + 0.0082111028105765343 * (sin(x[3]) * cos(x[3])) + 0.21830873966515735 * (sin(x[3]) * pow(cos(x[3]), 2)) - 7.1311711586800999e-06 * pow(x[1], 3) - 1.1491345926482223e-06 * pow(x[5], 3) + 5.290522284224498e-06 * pow(x[7], 3) + 0.40081340811141203 * pow(sin(x[3]), 3) + 1.0462160804427931 * sin(x[3])) * ((-1 * cos(x[3])) / (0.90000000000000002 * cos(x[2])))))) / 0.90000000000000002) * cos(x[3])) + 10.9 * sin(x[3])) / cos(x[2]))])
        Taylor3_SDSOS_Q2e4 = np.array([x[4],
                         x[5],
                         x[6],
                         x[7],
                         (0.025985759659187119 * x[0] + 158.07994174502926 * x[2] - 0.3716336569325065 * x[4] + 6.4587701771726376e-06 * x[5] + 7.5341973451989288 * x[6] + 1.8081490240376541e-05 * x[7] + 8.6101854355325684e-05 * (x[0] * x[1] * x[5]) - 2.2362441675895609e-05 * (x[0] * x[2] * x[4]) + 2.0126197508306046e-05 * (x[0] * x[2] * x[6]) - 1.0294480449040951e-05 * (x[1] * x[2] * x[3]) - 2.219845220968448e-05 * (x[1] * x[2] * x[5]) + 8.8009424710867011e-05 * (x[1] * x[4] * x[5]) - 0.00046710617642045367 * (x[2] * x[3] * x[4]) - 0.0002945314534245098 * (x[2] * x[3] * x[5]) - 0.00013254196247309175 * (x[2] * x[3] * x[6]) + 0.013781653511377088 * (x[2] * x[4] * x[6]) - 0.014349195069241701 * (x[3] * x[4] * x[7]) + 0.00017631162648910149 * (x[3] * x[6] * x[7]) + ((-163.21590736129596 * x[2] * pow(x[3], 2)) / 2) + ((-156.93901379606697 * pow(x[2], 3)) / 6) + ((-8.891341671468707 * pow(x[3], 2) * x[6]) / 2) + ((-1.5140437387672692 * pow(x[3], 2) * x[4]) / 2) + ((-0.46981419348525333 * pow(x[4], 3)) / 6) + ((-0.056986517831026637 * x[2] * pow(x[6], 2)) / 2) + ((-0.028514557695434106 * pow(x[1], 2) * x[2]) / 2) + ((-0.028120879944272436 * pow(x[0], 2) * x[2]) / 2) + ((-0.025090692738814192 * x[0] * pow(x[2], 2)) / 2) + ((-0.016092595799421284 * x[2] * pow(x[5], 2)) / 2) + ((-0.015066799039547983 * x[2] * pow(x[4], 2)) / 2) + ((-0.0073192793988367212 * x[2] * pow(x[7], 2)) / 2) + ((-0.0021314436129659439 * x[4] * pow(x[5], 2)) / 2) + ((-0.00040383728288752351 * pow(x[1], 2) * x[4]) / 2) + ((-0.00014423248254671469 * pow(x[0], 2) * x[4]) / 2) + ((-1.0763431740226093e-05 * pow(x[3], 2) * x[7]) / 2) + ((-6.4587701771726376e-06 * pow(x[2], 2) * x[5]) / 2) + ((-6.4587701771726376e-06 * pow(x[3], 2) * x[5]) / 2) + ((7.3180585001504433e-06 * pow(x[2], 2) * x[7]) / 2) + ((8.6355221724694915e-05 * x[0] * pow(x[5], 2)) / 2) + ((8.7588547593900578e-05 * x[0] * pow(x[7], 2)) / 2) + ((0.0012289624398264913 * x[0] * pow(x[4], 2)) / 2) + ((0.0015211431668420085 * pow(x[2], 2) * x[6]) / 2) + ((0.011308141515977189 * x[0] * pow(x[3], 2)) / 2) + ((0.029388560621793 * pow(x[2], 2) * x[4]) / 2)),
                         (0.028881439833938614 * x[1] + 164.31606006519559 * x[3] + 6.4587701771726376e-06 * x[4] - 0.36369515692380849 * x[5] + 1.8081490240376541e-05 * x[6] + 6.4062698640100599 * x[7] - 8.791042233320671e-06 * (x[0] * x[1] * x[2]) + 8.6101854355325684e-05 * (x[0] * x[1] * x[4]) - 0.00011673340560498513 * (x[0] * x[2] * x[3]) + 8.6355221724694915e-05 * (x[0] * x[4] * x[5]) - 9.7320608437667311e-05 * (x[0] * x[4] * x[7]) + 1.9978606988716032e-05 * (x[1] * x[2] * x[6]) - 0.00013747195370884162 * (x[1] * x[5] * x[7]) - 1.5100381917224561 * (x[2] * x[3] * x[4]) - 0.0005104843691054417 * (x[2] * x[3] * x[5]) - 8.8929666394410578 * (x[2] * x[3] * x[6]) - 1.8081490240376538e-05 * (x[2] * x[3] * x[7]) + 2.9785274336034646e-05 * (x[2] * x[4] * x[5]) + 0.014510142966381586 * (x[2] * x[5] * x[6]) - 0.0073192793988367212 * (x[2] * x[6] * x[7]) - 0.015541783691952376 * (x[3] * x[5] * x[7]) + ((-657.96028408747316 * pow(x[3], 3)) / 6) + ((-474.91496247076827 * pow(x[2], 2) * x[3]) / 2) + ((-6.4092722039791701 * pow(x[2], 2) * x[7]) / 2) + ((-6.4044892704425642 * pow(x[3], 2) * x[7]) / 2) + ((-1.7433342425888421 * pow(x[3], 2) * x[5]) / 2) + ((-0.53142479916341423 * pow(x[5], 3)) / 6) + ((-0.024950524189713351 * x[1] * pow(x[2], 2)) / 2) + ((-0.0021314436129659439 * pow(x[4], 2) * x[5]) / 2) + ((-0.00045855276166194248 * pow(x[0], 2) * x[5]) / 2) + ((-0.00030993484769751645 * pow(x[1], 3)) / 6) + ((-9.6422026586060819e-05 * pow(x[1], 2) * x[5]) / 2) + ((-4.9387366140279154e-05 * x[5] * pow(x[7], 2)) / 2) + ((-3.7383833396900723e-05 * pow(x[5], 2) * x[7]) / 2) + ((-1.0763431740226098e-05 * pow(x[3], 2) * x[6]) / 2) + ((-6.4587701771726376e-06 * pow(x[2], 2) * x[4]) / 2) + ((-6.4587701771726376e-06 * pow(x[3], 2) * x[4]) / 2) + ((1.3190444673823373e-05 * pow(x[0], 2) * x[1]) / 2) + ((2.5399548740526985e-05 * pow(x[2], 2) * x[6]) / 2) + ((7.472953457774082e-05 * pow(x[7], 3)) / 6) + ((8.8009424710867011e-05 * x[1] * pow(x[4], 2)) / 2) + ((0.00012372475833795745 * x[1] * pow(x[7], 2)) / 2) + ((0.00017631162648910149 * x[3] * pow(x[6], 2)) / 2) + ((0.0033402252575842608 * x[1] * pow(x[5], 2)) / 2) + ((0.0086970010215202988 * x[3] * pow(x[7], 2)) / 2) + ((0.010830936712579061 * x[1] * pow(x[3], 2)) / 2) + ((0.015943550076935223 * x[3] * pow(x[4], 2)) / 2) + ((0.017268648546613752 * x[3] * pow(x[5], 2)) / 2) + ((0.018765305007514912 * pow(x[1], 2) * x[3]) / 2) + ((0.01938138365082779 * pow(x[0], 2) * x[3]) / 2) + ((0.036790575236646472 * pow(x[2], 2) * x[5]) / 2)),
                         ( - 0.028873066287985687 * x[0] - 164.74437971669917 * x[2] + 0.41292628548056276 * x[4] - 7.1764113079695978e-06 * x[5] - 8.3713303835543655 * x[6] - 2.0090544711529489e-05 * x[7] - 9.5668727061472977e-05 * (x[0] * x[1] * x[5]) + 2.4847157417661789e-05 * (x[0] * x[2] * x[4]) - 2.2362441675895609e-05 * (x[0] * x[2] * x[6]) + 0.032101927015986284 * (x[1] * x[2] * x[3]) + 2.4664946899649425e-05 * (x[1] * x[2] * x[5]) - 9.7788249678741122e-05 * (x[1] * x[4] * x[5]) + 0.00052618327399736251 * (x[2] * x[3] * x[4]) - 0.40377847274487111 * (x[2] * x[3] * x[5]) + 0.00016735939190385364 * (x[2] * x[3] * x[6]) + 7.1180776266778443 * (x[2] * x[3] * x[7]) - 0.015312948345974544 * (x[2] * x[4] * x[6]) + 0.015943550076935223 * (x[3] * x[4] * x[7]) - 0.00019590180721011281 * (x[3] * x[6] * x[7]) + ((-1.9918674673346259 * x[2] * pow(x[7], 2)) / 2) + ((-0.44558024172699939 * pow(x[2], 2) * x[4]) / 2) + ((-0.012564601684419098 * x[0] * pow(x[3], 2)) / 2) + ((-0.0013655138220294348 * x[0] * pow(x[4], 2)) / 2) + ((-9.7320608437667311e-05 * x[0] * pow(x[7], 2)) / 2) + ((-9.5950246360772143e-05 * x[0] * pow(x[5], 2)) / 2) + ((7.176411307969597e-06 * pow(x[3], 2) * x[5]) / 2) + ((1.1959368600251213e-05 * pow(x[3], 2) * x[7]) / 2) + ((1.1959368600251218e-05 * pow(x[2], 2) * x[7]) / 2) + ((1.4352822615939194e-05 * pow(x[2], 2) * x[5]) / 2) + ((0.00016025831394079408 * pow(x[0], 2) * x[4]) / 2) + ((0.00044870809209724837 * pow(x[1], 2) * x[4]) / 2) + ((0.0023682706810732711 * x[4] * pow(x[5], 2)) / 2) + ((0.01674088782171998 * x[2] * pow(x[4], 2)) / 2) + ((0.017880661999356982 * x[2] * pow(x[5], 2)) / 2) + ((0.031245422160302704 * pow(x[0], 2) * x[2]) / 2) + ((0.03168284188381567 * pow(x[1], 2) * x[2]) / 2) + ((0.056751613775557014 * x[0] * pow(x[2], 2)) / 2) + ((0.063318353145585149 * x[2] * pow(x[6], 2)) / 2) + ((0.52201577053917037 * pow(x[4], 3)) / 6) + ((1.6822708208525212 * pow(x[3], 2) * x[4]) / 2) + ((8.3696402244800971 * pow(x[2], 2) * x[6]) / 2) + ((9.8792685238541189 * pow(x[3], 2) * x[6]) / 2) + ((535.59780832409683 * x[2] * pow(x[3], 2)) / 2) + ((690.40982114572751 * pow(x[2], 3)) / 6)),
                         ( - 0.032090488704376238 * x[1] - 171.67340007243953 * x[3] - 7.1764113079695978e-06 * x[4] + 0.40410572991534277 * x[5] - 2.0090544711529489e-05 * x[6] - 7.1180776266778443 * x[7] + 9.7678247036896333e-06 * (x[0] * x[1] * x[2]) - 9.5668727061472977e-05 * (x[0] * x[1] * x[4]) + 0.00012970378400553901 * (x[0] * x[2] * x[3]) - 9.5950246360772129e-05 * (x[0] * x[4] * x[5]) + 0.00010813400937518589 * (x[0] * x[4] * x[7]) - 2.2198452209684476e-05 * (x[1] * x[2] * x[6]) + 0.00015274661523204623 * (x[1] * x[5] * x[7]) + 1.6778202130249511 * (x[2] * x[3] * x[4]) + 0.00056720485456160179 * (x[2] * x[3] * x[5]) + 9.8810740438233964 * (x[2] * x[3] * x[6]) + 2.0090544711529486e-05 * (x[2] * x[3] * x[7]) - 3.3094749262260714e-05 * (x[2] * x[4] * x[5]) - 0.016122381073757314 * (x[2] * x[5] * x[6]) + 2.0081325326653743 * (x[2] * x[6] * x[7]) + 0.017268648546613748 * (x[3] * x[5] * x[7]) + ((-0.021534870723141988 * pow(x[0], 2) * x[3]) / 2) + ((-0.02085033889723879 * pow(x[1], 2) * x[3]) / 2) + ((-0.019187387274015279 * x[3] * pow(x[5], 2)) / 2) + ((-0.017715055641039135 * x[3] * pow(x[4], 2)) / 2) + ((-0.0096633344683558862 * x[3] * pow(x[7], 2)) / 2) + ((-0.0043676840491391825 * x[1] * pow(x[2], 2)) / 2) + ((-0.003711361397315845 * x[1] * pow(x[5], 2)) / 2) + ((-0.00019590180721011275 * x[3] * pow(x[6], 2)) / 2) + ((-0.00013747195370884159 * x[1] * pow(x[7], 2)) / 2) + ((-9.7788249678741108e-05 * x[1] * pow(x[4], 2)) / 2) + ((-8.3032816197489804e-05 * pow(x[7], 3)) / 6) + ((-4.8312265534337242e-05 * pow(x[2], 2) * x[6]) / 2) + ((-1.4656049637581524e-05 * pow(x[0], 2) * x[1]) / 2) + ((-1.6940658945086007e-21 * pow(x[2], 2) * x[4]) / 2) + ((1.4352822615939194e-05 * pow(x[3], 2) * x[4]) / 2) + ((3.2049913311780707e-05 * pow(x[3], 2) * x[6]) / 2) + ((4.1537592663223022e-05 * pow(x[5], 2) * x[7]) / 2) + ((5.4874851266976834e-05 * x[5] * pow(x[7], 2)) / 2) + ((0.00010713558509562313 * pow(x[1], 2) * x[5]) / 2) + ((0.00034437205299724046 * pow(x[1], 3)) / 6) + ((0.0005095030685132694 * pow(x[0], 2) * x[5]) / 2) + ((0.0023682706810732707 * pow(x[4], 2) * x[5]) / 2) + ((0.0033359332990112733 * pow(x[2], 2) * x[7]) / 2) + ((0.020056114579288395 * x[1] * pow(x[3], 2)) / 2) + ((0.36322731298573557 * pow(x[2], 2) * x[5]) / 2) + ((0.59047199907046022 * pow(x[5], 3)) / 6) + ((1.5329323174055929 * pow(x[3], 2) * x[5]) / 2) + ((14.23417681605847 * pow(x[3], 2) * x[7]) / 2) + ((356.00989156174745 * pow(x[2], 2) * x[3]) / 2) + ((1267.887182536733 * pow(x[3], 3)) / 6))])

        return Taylor3_SDSOS_Q2e4


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
