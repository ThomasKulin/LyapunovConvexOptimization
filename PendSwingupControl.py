import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
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
from pydrake.examples import PendulumParams
from scipy.integrate import quad

from underactuated import ConfigureParser

meshcat = StartMeshcat()

if "MOSEKLM_LICENSE_FILE" not in os.environ:
    # If a mosek.lic file has already been uploaded, then simply use it here.
    if os.path.exists('/tmp/mosek.lic'):
        os.environ["MOSEKLM_LICENSE_FILE"] = "/tmp/mosek.lic"
    else:
        print("stop bein a loooooser and get mosek")

print(MosekSolver().enabled())

#System dimensions. Here:
# x = [theta, theta_dot]
# z = [sin(theta), cos(theta), theta_dot]
nx = 2
nz = 3
nu = 1

# Map from original state to augmented state.
# Uses sympy to be able to do symbolic integration later on.
x2z = lambda x: np.array([np.sin(x[0]), np.cos(x[0]), x[1]])

# System dynamics in augmented state (z).
params = PendulumParams()
inertia = params.mass() * params.length() ** 2
tau_g = params.mass() * params.gravity() * params.length()


def f(z, u):
    return [
        z[1] * z[2],
        -z[0] * z[2],
        (tau_g * z[0] + u[0] - params.damping() * z[2]) / inertia,
    ]


# State limits (region of state space where we approximate the value function).
x_max = np.array([np.pi, 2 * np.pi])
x_min = -x_max
z_max = np.array([1, 1, x_max[-1]])
z_min = -z_max
x_max_int = np.array([np.pi, np.pi])
z_max_int = np.array([1, 1, x_max_int[-1]])
z_min_int = -z_max_int
u_max = np.array([1.8])

# Equilibrium point in both the system coordinates.
x0 = np.array([0, 0])
z0 = x2z(x0)

# Quadratic running cost in augmented state.
Q_diag = np.ones(nz) * 5
Q = np.diag(Q_diag)
R = np.diag([1])
Rinv = np.linalg.inv(R)


def l(z, u):
    return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)

# Given the degree for the approximate value function and the polynomials
# in the S procedure, solves the SOS and returns the approximate value function
# (together with the objective of the SOS program).
def pendulum_sos_dp(deg):
    f2 = np.array([[0], [0], [1 / inertia]])

    # Set up optimization.
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    u = prog.NewIndeterminates(nu, "u")
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    # Maximize volume beneath the value function, integrating over the ring
    # s^2 + c^2 = 1.
    obj = J.Integrate(z[-1], z_min_int[-1], z_max_int[-1])
    cost = 0
    for monomial, coeff in obj.monomial_to_coefficient_map().items():
        s_deg = monomial.degree(z[0])
        c_deg = monomial.degree(z[1])
        monomial_int = quad(
            lambda x: np.sin(x) ** s_deg * np.cos(x) ** c_deg, 0, 2 * np.pi
        )[0]
        cost += monomial_int * coeff
    poly = Polynomial(cost)
    cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
    # Make the numerics better
    cost = (
        Polynomial(cost / np.max(np.abs(cost_coeff)))
        .RemoveTermsWithSmallCoefficients(1e-6)
        .ToExpression()
    )
    prog.AddLinearCost(-cost)

    J_dot = J_expr.Jacobian(z).dot(f(z, u))
    LHS = J_dot + l(z, u)

    # S procedure for s^2 + c^2 = 1.
    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[0] ** 2 + z[1] ** 2 - 1)
    S_Jdot = 0
    for i in range(nz):
        lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(deg / 2) * 2))[
            0
        ].ToExpression()
        S_Jdot += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])

    # Enforce Input constraint
    u_min = -u_max
    for i in range(nu):
        lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(deg / 2) * 2))[
            0
        ].ToExpression()
        S_Jdot += lam * (u[i] - u_max[i]) * (u[i] - u_min[i])
    # Enforce Bellman inequality.
    prog.AddSosConstraint(LHS + S_r + S_Jdot)

    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[0] ** 2 + z[1] ** 2 - 1)
    S_J = 0
    for i in range(nz):
        lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(deg / 2) * 2))[
            0
        ].ToExpression()
        S_J += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])
    # Enforce that value function is PD
    prog.AddSosConstraint(J_expr + S_r + S_J)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Solve and retrieve result.
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = Solve(prog)

    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(
        1e-6
    )

    # Solve for the optimal feedback in augmented coordinates.
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = -0.5 * Rinv.dot(f2.T).dot(dJdz.T)

    return J_star, u_star, z


J_star, u_star, z = pendulum_sos_dp(deg=6)

X1, X2 = np.meshgrid(
    np.linspace(x_min[0], x_max[0], 51), np.linspace(x_min[1], x_max[1], 51)
)
X = np.vstack((X1.flatten(), X2.flatten()))
Z = x2z(X)
J = np.zeros(Z.shape[1])
for i in range(Z.shape[1]):
    J[i] = J_star.Evaluate({z[0]: Z[0, i], z[1]: Z[1, i], z[2]: Z[2, i]})

fig = plt.figure(figsize=(9, 4))
ax = fig.subplots()
ax.set_xlabel("q")
ax.set_ylabel("qdot")
ax.set_title("Cost-to-Go")
ax.imshow(
    J.reshape(X1.shape),
    cmap=cm.jet,
    aspect="auto",
    extent=(x_min[0], x_max[0], x_min[1], x_max[1]),
)
ax.invert_yaxis()
plt.show()