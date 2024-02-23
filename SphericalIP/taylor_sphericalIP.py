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
nz = 10
nq = 4
nx = 2 * nq
nu = 2

mp = 90
l = 0.9
g = 9.81

# Map from original state to augmented state.
# Uses sympy to be able to do symbolic integration later on.
# x = (x, y, theta, phi, xdot, ydot, thetadot, phidot)
# z = (x, y, st, ct, sp, cp, xdot, ydot, thetadot, phidot)
x2z = lambda x: np.array([x[0], x[1], np.sin(x[2]), np.cos(x[2]), np.sin(x[3]), np.cos(x[3]), x[4], x[5], x[6], x[7]])

# Transform to augmented representation
def T(z, dtype=Expression):
    assert len(z) == nz
    T = np.zeros([nz, nx], dtype=dtype)
    T[0, 0] = 1
    T[1, 1] = 1
    T[2, 2] = z[3]
    T[3, 2] = -z[2]
    T[4, 3] = z[5]
    T[5, 3] = -z[4]
    T[6, 4] = 1
    T[7, 5] = 1
    T[8, 6] = 1
    T[9, 7] = 1
    return T

# z = (x, y, st, ct, sp, cp, xdot, ydot, thetadot, phidot)
# d
def f(z, u, T, dtype=Expression):
    st = z[2]
    ct = z[3]
    sp = z[4]
    cp = z[5]
    theta_dot = z[8]
    phi_dot = z[9]

    assert len(z) == nz
    assert len(u) == nu
    if dtype == float:
        assert (st ** 2 + ct ** 2 * sp ** 2 + cp ** 2 * ct ** 2) == 1

    qdot = z[-nq:]
    f_val = np.zeros(nx, dtype=Expression)
    f_val[:nq] = qdot
    f_val[4] = u[0]  # xddot
    f_val[5] = u[1]  # yddot
    f_val[6] = (g / l * cp * st - phi_dot ** 2 * ct * st - u[0] / l * ct + u[1] / l * sp * st)  # thetaddot
    f_val[7] = (g / l * sp + 2 * phi_dot * theta_dot * st - u[1] / l * cp)/ct # phiddot
    return T @ f_val

def fx(x, u, dtype=Expression):
    st = sin(x[2])
    ct = cos(x[2])
    sp = sin(x[3])
    cp = cos(x[3])
    theta_dot = x[6]
    phi_dot = x[7]
    qdot = x[-nq:]
    f_val = np.zeros(nx, dtype=Expression)
    f_val[:nq] = qdot
    f_val[4] = u[0]  # xddot
    f_val[5] = u[1]  # yddot
    f_val[6] = (g / l * cp * st - phi_dot ** 2 * ct * st - u[0] / l * ct + u[1] / l * sp * st)  # thetaddot
    f_val[7] = (g / l * sp + 2 * phi_dot * theta_dot * st - u[1] / l * cp)/ct # phiddot
    return f_val

def f2(z, T, dtype=Expression):
    assert len(z) == nz
    st = z[2]
    ct = z[3]
    sp = z[4]
    cp = z[5]
    f2_val = np.zeros([nx, nu], dtype=dtype)
    f2_val[4, :] = [1, 0]
    f2_val[5, :] = [0, 1]
    f2_val[6, :] = [-cp / l, sp * st / l]
    f2_val[7, :] = [0, -cp / (l * ct)]
    return T @ f2_val

def saveSystem_py(f_val, filepath, nx):
    with open(filepath, "w") as text_file:
        for f in f_val:
            text = f.__str__()
            for i in range(nx):
                text = text.replace(f"x({i})", f"x[{i}]")
            text_file.write(text + "\n")

def saveSystem_mat(f_val, filepath, nx):
    with (open(filepath, "w") as text_file):
        for f in f_val:
            text = f.__str__().replace("x(0)", "x") \
            .replace("x(1)", "y") \
            .replace("x(2)", "theta") \
            .replace("x(3)", "phi") \
            .replace("x(4)", "x_dot") \
            .replace("x(5)", "y_dot") \
            .replace("x(6)", "theta_dot") \
            .replace("x(7)", "phi_dot")
            text = re.sub(r"pow\(([^,]+),\s*(\d+)\)", r"\1^\2", text)
            text_file.write(text + "\n")
            
def zToX(expr, z, x):
    for i in range(len(expr)):
        expr[i] = expr[i].Substitute(z[0], x[0])
        expr[i] = expr[i].Substitute(z[1], x[1])
        expr[i] = expr[i].Substitute(z[2], sin(x[2]))
        expr[i] = expr[i].Substitute(z[3], cos(x[2]))
        expr[i] = expr[i].Substitute(z[4], sin(x[3]))
        expr[i] = expr[i].Substitute(z[5], cos(x[3]))
        expr[i] = expr[i].Substitute(z[6], x[4])
        expr[i] = expr[i].Substitute(z[7], x[5])
        expr[i] = expr[i].Substitute(z[8], x[6])
        expr[i] = expr[i].Substitute(z[9], x[7])
    return expr


prog = MathematicalProgram()


z = prog.NewIndeterminates(nz, "z")  # shifted system state
u = prog.NewIndeterminates(nu, "u")
# Equilibrium point in both the system coordinates.
# x = (x, y, theta, phi, xdot, ydot, thetadot, phidot)
x0 = np.zeros(nx)
z0 = x2z(x0)
z0[np.abs(z0) <= 1e-6] = 0


# Define Lyapunov function from optimization
# filepath = "./data/[1.5 1.5 1.  1.  1.  1.  4.  4.  3.  3. ]/kSdsos/"
# filename = "J_lower_bound_deg_4_TP2e4_TPDOT2e2.pkl"
filepath = "./data/[1.5 1.5 1.  1.  1.  1.  4.  4.  3.  3. ]/"
filename = "J_lower_bound_deg_4_SDSOS_Q20000.pkl"
J_star = load_polynomial(z, filepath+filename)
R = np.array([[0.1, 0.0], [0.0, 0.1]])
Rinv = np.linalg.inv(R)
T_val = T(z)
f2_val = f2(z, T_val)
dJdz = J_star.Jacobian(z)
u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)

x = prog.NewIndeterminates(nx, "x")


# f_val= f(z, u_star, T_val)
# for i in range(len(f_val)):
#     f_val[i] = f_val[i].Substitute(z[0], x[0])
#     f_val[i] = f_val[i].Substitute(z[1], x[1])
#     f_val[i] = f_val[i].Substitute(z[2], sin(x[2]))
#     f_val[i] = f_val[i].Substitute(z[3], cos(x[2]))
#     f_val[i] = f_val[i].Substitute(z[4], sin(x[3]))
#     f_val[i] = f_val[i].Substitute(z[5], cos(x[3]))
#     f_val[i] = f_val[i].Substitute(z[6], x[4])
#     f_val[i] = f_val[i].Substitute(z[7], x[5])
#     f_val[i] = f_val[i].Substitute(z[8], x[6])
#     f_val[i] = f_val[i].Substitute(z[9], x[7])
# zToX(f_val, z, z)

# for i in range(len(u_star)):
#     u_star[i] = u_star[i].Substitute(z[0], x[0])
#     u_star[i] = u_star[i].Substitute(z[1], x[1])
#     u_star[i] = u_star[i].Substitute(z[2], sin(x[2]))
#     u_star[i] = u_star[i].Substitute(z[3], cos(x[2]))
#     u_star[i] = u_star[i].Substitute(z[4], sin(x[3]))
#     u_star[i] = u_star[i].Substitute(z[5], cos(x[3]))
#     u_star[i] = u_star[i].Substitute(z[6], x[4])
#     u_star[i] = u_star[i].Substitute(z[7], x[5])
#     u_star[i] = u_star[i].Substitute(z[8], x[6])
#     u_star[i] = u_star[i].Substitute(z[9], x[7])



# save taylor approximation in python format
u_star = zToX(u_star, z, x)
f_val= fx(x, u_star)
order = 3
# f_taylor = [TaylorExpand(f, {x[i]: x0[i] for i in range(len(x))}, order) for f in f_val]
# filename = filename+f"_TaylorSeries_Order{order}.txt"
# saveSystem_py(f_taylor, filepath+filename, nx)

# Save the full system dymanics in a Matlab Readable Format
# filename = filename+f"_closedloop.txt"
# saveSystem_mat(f_val, filepath+filename, nx)

# save J_star in python format
J_star_x = zToX([J_star], z, x)
J_taylor = [TaylorExpand(f, {x[i]: x0[i] for i in range(len(x))}, order) for f in J_star_x]
filename = filename+f"_Jstar.txt"
saveSystem_py(J_taylor, filepath+filename, nx)



print("done")
