import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

# pydrake imports
from pydrake.all import (LinearQuadraticRegulator, MathematicalProgram,
                         RealContinuousLyapunovEquation, MonomialBasis, Solve, Polynomial, Variables, Monomial)
from pydrake.examples import VanDerPolOscillator
# underactuated imports
from underactuated import plot_2d_phase_portrait
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from matplotlib import cm
from utils import calc_u_opt , save_polynomial, load_polynomial
from pydrake.symbolic import sin, cos
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
    Linearize,
    Jacobian,
    RegionOfAttraction,
    LinearQuadraticRegulator,
    CommonSolverOption,
    DiagramBuilder,
    Expression,
    LeafSystem,
    LogVectorOutput,
    MathematicalProgram,
    MeshcatVisualizer,
    MosekSolver,
    Parser,
    Polynomial,
    Simulator,
    Solve,
    SolverOptions,
    StartMeshcat,
    SymbolicVectorSystem,
    TaylorExpand,
    Variable,
    Variables,
    WrapToSystem,
    to_sympy,
    from_sympy,
)

mc = 10
mp = 1
l = .5
g = 9.81

nz = 5
nq = 2
nx = 2 * nq
nu = 1

# Map from original state to augmented state.
# Uses sympy to be able to do symbolic integration later on.
# x = (x, theta, xdot, thetadot)
# z = (x, s, c, xdot, thetadot)
x2z = lambda x: np.array([x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])

def T(z, dtype=Expression):
    assert len(z) == nz
    T = np.zeros([nz, nx], dtype=dtype)
    T[0, 0] = 1
    T[1, 1] = z[2]
    T[2, 1] = -z[1]
    T[3, 2] = 1
    T[4, 3] = 1
    return T

def f(x, u):
    s = np.sin(x[1])
    c = np.cos(x[1])
    qdot = x[-nq:]
    f_val = np.zeros(nx, dtype=Expression)
    f_val[:nq] = qdot
    f_val[2] = ((u + mp * s * (l * qdot[1] ** 2 + g * c))/(mc + mp * s ** 2))[0]
    f_val[3] = ((-u * c - mp * l * qdot[1] ** 2 * c * s - (mc + mp) * g * s) / ((mc + mp * s ** 2)*l))[0]
    return f_val

def f2(z, T, dtype=Expression):
    assert len(z) == nz
    s = z[1]
    c = z[2]
    f2_val = np.zeros([nx, nu], dtype=dtype)
    denom = (mc + mp * s ** 2)
    f2_val[2, :] = 1/denom
    f2_val[3, :] = -c / l /denom
    return T @ f2_val

def saveSystem(f_taylor, filepath, nx):
    with open(filepath, "w") as text_file:
        for f in f_taylor:
            text = f.__str__()
            for i in range(nx):
                text = text.replace(f"x({i})", f"x[{i}]")
            text_file.write(text + "\n")

prog = MathematicalProgram()

# x = (x, theta, xdot, thetadot)
# z = (x, s, c, xdot, thetadot)
z = prog.NewIndeterminates(nz, "z")  # shifted system state
u = prog.NewIndeterminates(nu, "u")
x0 = np.array([0, np.pi, 0, 0])
z0 = x2z(x0)
zu = np.concatenate((z, u))


# Define Lyapunov function from optimization
# filename = "/home/thomas/Documents/thesis/LyapunovConvexOptimization/SphericalIP/data/test/J_upper_bound_lower_deg_1_deg_2.pkl"
filename = "/home/thomas/Documents/thesis/LyapunovConvexOptimization/SphericalIP/data/test/J_lower_bound_deg_6.pkl"
J_star = load_polynomial(z, filename)
R = np.diag([1])
Rinv = np.linalg.inv(R)
T_val = T(z)
f2_val = f2(z, T_val)
dJdz = J_star.Jacobian(z)
u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)



x = prog.NewIndeterminates(nx, "x")

u_star[0] = u_star[0].Substitute(z[0], x[0])
u_star[0] = u_star[0].Substitute(z[1], sin(x[1]))
u_star[0] = u_star[0].Substitute(z[2], cos(x[1]))
u_star[0] = u_star[0].Substitute(z[3], x[2])
u_star[0] = u_star[0].Substitute(z[4], x[3])

order = 3
f_val= f(x, u_star)
f_taylor = [TaylorExpand(f, {x[i]: x0[i] for i in range(len(x))}, order) for f in f_val]

filepath = f'/home/thomas/Documents/thesis/LyapunovConvexOptimization/CartPole_2D/TaylorSeries_Order{order}.txt'
saveSystem(f_taylor,filepath, nx)

print("done")
