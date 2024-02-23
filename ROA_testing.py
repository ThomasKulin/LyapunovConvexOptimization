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
def f(z, u, u_denom, T, dtype=Expression):
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
    f_val[4] = u[0] * denominator * u_denom # xddot
    f_val[5] = u[1] * denominator * u_denom # yddot
    f_val[6] = ((g / l * cp * st - phi_dot ** 2 * ct * st)*u_denom - u[0] / l * ct + u[1] / l * sp * st) * denominator  # thetaddot
    f_val[7] = ((g / l * sp + 2 * phi_dot * theta_dot * st)*u_denom - u[1] / l * cp) # phiddot
    return T @ f_val, denominator*u_denom

def f2(z, T, dtype=Expression):
    assert len(z) == nz
    st = z[2]
    ct = z[3]
    sp = z[4]
    cp = z[5]
    f2_val = np.zeros([nx, nu], dtype=dtype)
    denom = ct
    f2_val[4, :] = [1*denom, 0]
    f2_val[5, :] = [0, 1*denom]
    f2_val[6, :] = [-cp / l *denom, sp * st / l *denom]
    f2_val[7, :] = [0, -cp / (l * 1)]
    return T @ f2_val, denom

prog = MathematicalProgram()

z = prog.NewIndeterminates(nz, "z")
u = prog.NewIndeterminates(nu, "u")
zu = np.concatenate((z, u))
x0 = np.zeros(nx)
z0 = x2z(x0)
z0[np.abs(z0) <= 1e-6] = 0

# Define Lyapunov function from optimization
filename = "SphericalIP/data/[1.5 1.5 1.  1.  1.  1.  4.  4.  3.  3. ]/kSdsos/J_lower_bound_deg_2.pkl"
J_star = load_polynomial(z, filename)
R = np.array([[0.1, 0.0], [0.0, 0.1]])
Rinv = np.linalg.inv(R)
T_val = T(z)
f2_val, u_denom = f2(z, T_val)
dJdz = J_star.Jacobian(z)
u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)
u_star[0] = Polynomial(u_star[0]).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
u_star[1] = Polynomial(u_star[1]).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
f_val, denom = f(z, u_star, u_denom, T_val)


V = J_star
V = Polynomial(J_star).RemoveTermsWithSmallCoefficients(1e-0)
V_dot = dJdz.dot(f_val)
V_dot = Polynomial(V_dot).RemoveTermsWithSmallCoefficients(1e-0)

"""Verify that J is a Lyapunov function close to the origin"""
# # small region to test
z_max = np.array([0.1, 0.1, 0.1, 1, 0.1, 1, 0.1, 0.1, 0.1, 0.1])
z_min = np.array([-0.1, -0.1, -0.1, 0.9, -0.1, 0, -0.1, -0.1, -0.1, -0.1])
#
# # S procedure for st^2 + ct^2 + sp^2 + cp^2 = 2.
# ring_deg = 2
# lam = prog.NewFreePolynomial(Variables(zu), ring_deg).ToExpression()  # doesnt have to be SOS!! bc we're dealing with "g"==0 not <=0
# S_sphere = lam * (z[2] ** 2 + z[3] ** 2 * z[4] ** 2 + z[5] ** 2 * z[3] ** 2 - 1)
# # S procedure for z_min < z < z_max
# S_Jdot = 0
# for i in np.arange(nz):
#     lam = prog.NewSosPolynomial(Variables(zu), ring_deg, type=prog.NonnegativePolynomial.kSdsos)[0].ToExpression()
#     S_Jdot += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])  # negative inside the range of z-space we are interested in
#
# VDOT_FEAS_EPS = 3e-5
# prog.AddSosConstraint(-V_dot - VDOT_FEAS_EPS*V + S_sphere + S_Jdot, type=prog.NonnegativePolynomial.kSdsos)
#
#
# # Solve and retrieve result.
# options = SolverOptions()
# options.SetOption(CommonSolverOption.kPrintToConsole, 1)
# prog.SetSolverOptions(options)
# mosek_available = MosekSolver().available() and MosekSolver().enabled()
# if not mosek_available:
#     Error("Mosek is not available. Skipping this example.")
# result = Solve(prog)


"""solve for the largest level set of J that represents the region of attraction"""
# small region to test
# z_max = np.array([1.5, 1.5, np.sin(np.pi / 2), 1, np.sin(np.pi / 2), 1, 4, 4, 3, 3])
# z_min = np.array([-1.5, -1.5, -np.sin(np.pi / 2), 0, -np.sin(np.pi / 2), 0, -4, -4, -3, -3])

# S procedure for st^2 + ct^2 + sp^2 + cp^2 = 2.
ring_deg = 2
lam = prog.NewFreePolynomial(Variables(zu), ring_deg).ToExpression()  # doesnt have to be SOS!! bc we're dealing with "g"==0 not <=0
S_sphere = lam * (z[2] ** 2 + z[3] ** 2 * z[4] ** 2 + z[5] ** 2 * z[3] ** 2 - 1)
# S procedure for z_min < z < z_max
S_Jdot = 0
for i in np.arange(nz):
    lam = prog.NewSosPolynomial(Variables(zu), ring_deg, type=prog.NonnegativePolynomial.kSdsos)[0].ToExpression()
    S_Jdot += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])  # negative inside the range of z-space we are interested in

l_deg = 2
# l_poly, l_gram = prog.NewSosPolynomial(Variables(z), l_deg, type=prog.NonnegativePolynomial.kSdsos)
l_poly = prog.NewFreePolynomial(Variables(z), l_deg)
l = l_poly.ToExpression()
rho = prog.NewContinuousVariables(1)

z_normsq = (z-z0).dot(z-z0)
EPS = 3e-5
prog.AddSosConstraint((z_normsq * (V - rho[0]) + l * (-V_dot - EPS * V)) + S_sphere + S_Jdot, type=prog.NonnegativePolynomial.kSdsos)
# prog.AddSosConstraint(-V_dot - EPS*z_normsq + l*(V - rho[0]) + S_sphere + S_Jdot, type=prog.NonnegativePolynomial.kSdsos)

prog.AddLinearCost(-rho[0])
# prog.AddConstraint(rho[0] >= 0)

# Solve and retrieve result.
options = SolverOptions()
options.SetOption(CommonSolverOption.kPrintToConsole, 1)
prog.SetSolverOptions(options)
mosek_available = MosekSolver().available() and MosekSolver().enabled()
if not mosek_available:
    Error("Mosek is not available. Skipping this example.")
result = Solve(prog)


rho_sol = result.GetSolution(rho)[0]

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