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
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
    Linearize,
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
    Variable,
    Variables,
    WrapToSystem,
)

if "MOSEKLM_LICENSE_FILE" not in os.environ:
    # If a mosek.lic file has already been uploaded, then simply use it here.
    if os.path.exists('/home/thomas/mosek/mosek.lic'):
        os.environ["MOSEKLM_LICENSE_FILE"] = "/home/thomas/mosek/mosek.lic"
    else:
        print("stop bein a loooooser and get mosek")

print(MosekSolver().enabled())

# increase default size matplotlib figures
rcParams['figure.figsize'] = (6, 6)

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
    denominator = ct
    f_val = np.zeros(nx, dtype=Expression)
    f_val[:nq] = qdot * denominator
    f_val[4] = u[0] * denominator  # xddot
    f_val[5] = u[1] * denominator  # yddot
    f_val[6] = (g / l * cp * st - phi_dot ** 2 * ct * st - u[0] / l * ct + u[1] / l * sp * st) * denominator  # thetaddot
    f_val[7] = (g / l * sp + 2 * phi_dot * theta_dot * st - u[1] / l * cp) # phiddot
    return T @ f_val, denominator

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


prog = MathematicalProgram()

z = prog.NewIndeterminates(nz, "z")
u = prog.NewIndeterminates(nu, "u")

# Define Lyapunov function from optimization
filename = "/home/thomas/Documents/LyapunovConvexOptimization/SphericalIP/data/[1.5 1.5 1.  1.  1.  1.  4.  4.  3.  3. ]/J_lower_bound_deg_4_Q20000.pkl"
J_star = load_polynomial(z, filename)
R = np.array([[0.1, 0.0], [0.0, 0.1]])
Rinv = np.linalg.inv(R)
T_val = T(z)
f_val, denom = f(z, u, T_val)
f2_val = f2(z, T_val)
dJdz = J_star.Jacobian(z)
u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)


# V = x.dot(P).dot(x)
# V_dot = 2 * x.dot(P).dot(f(x))
V = J_star
V_dot = dJdz.dot(f_val)

l_deg = 4
# l_poly, l_gram = prog.NewSosPolynomial(Variables(z), l_deg, type=prog.NonnegativePolynomial.kSdsos)
l_poly = prog.NewFreePolynomial(Variables(z), l_deg)
l = l_poly.ToExpression()

rho = prog.NewContinuousVariables(1)

z_normsq = z.dot(z)
VDOT_FEAS_EPS = 3e-5
prog.AddSosConstraint((z_normsq * (V - rho[0]) + l * (V_dot + VDOT_FEAS_EPS * z_normsq)), type=prog.NonnegativePolynomial.kSdsos)

prog.AddLinearCost(-rho[0])

# Solve and retrieve result.
options = SolverOptions()
options.SetOption(CommonSolverOption.kPrintToConsole, 1)
prog.SetSolverOptions(options)
mosek_available = MosekSolver().available() and MosekSolver().enabled()
if not mosek_available:
    Error("Mosek is not available. Skipping this example.")
result = Solve(prog)
rho_sol = result.GetSolution(rho)[0]
# l_sol_gram = result.GetSolution(l_gram)

x_old = z

print(result.is_success())
print(f'rho = {rho_sol}.')



def plot_V(rho, label, color, linestyle):
    # grid of the state space
    x1 = np.linspace(*xlim)
    x2 = np.linspace(*xlim)
    X1, X2 = np.meshgrid(x1, x2)

    # function that evaluates V(x) at a given x
    # (looks bad, but it must accept meshgrids)
    eval_V = lambda x: sum(sum(x[i] * x[j] * Pij for j, Pij in enumerate(Pi)) for i, Pi in enumerate(P))

    # contour plot with only the rho level set
    cs = plt.contour(X1, X2, eval_V([X1, X2]), levels=[rho], colors=color, linestyle=linestyle, linewidths=3, zorder=3)

    # misc plot settings
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.gca().set_aspect('equal')

    # fake plot for legend
    plt.plot(0, 0, color=color, linewidth=3, label=label)
    plt.legend()
    return cs


# function that plots the levels sets of Vdot(x)
def plot_Vdot():
    # grid of the state space
    x1 = np.linspace(*xlim)
    x2 = np.linspace(*xlim)
    X1, X2 = np.meshgrid(x1, x2)

    # function that evaluates Vdot(x) at a given x
    eval_Vdot = lambda x: 2 * sum(sum(x[i] * f(x)[j] * Pij for j, Pij in enumerate(Pi)) for i, Pi in enumerate(P))

    # contour plot with only the rho level set
    cs = plt.contour(X1, X2, eval_Vdot([X1, X2]), colors='b', levels=np.linspace(-10, 40, 11))
    plt.gca().clabel(cs, inline=1, fontsize=10)

    # misc plot settings
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.gca().set_aspect('equal')

    # fake plot for legend
    plt.plot(0, 0, color='b', label=r'$\dot{V}(\mathbf{x})$')
    plt.legend()
    return cs


xlim = (-3, 3)
limit_cycle = VanDerPolOscillator.CalcLimitCycle()

plot_2d_phase_portrait(f, x1lim=xlim, x2lim=xlim)
plot_V(rho_sol, "SOS relaxed", "m", linestyle='dashed')
plt.plot(limit_cycle[0], limit_cycle[1], color='b', linewidth=3, label='ROA boundary')
plt.legend(loc=1)
plt.show()

fig, ax = plt.subplots()
plot_Vdot()
plot_V(rho_sol, "SOS relaxed", "m", linestyle='dashed')
plt.show()