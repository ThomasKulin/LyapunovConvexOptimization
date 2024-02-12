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


nz = 5
nq = 2
nx = 2 * nq
nu = 1

mc = 10
mp = 1
l = 0.5
g = 9.81

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

""" Analyzing f, f_val is the normal IP on cart eqns, with denominator kept separate"""
def f(z, u, T):
    assert len(z) == nz
    s = z[1]
    c = z[2]
    qdot = z[-nq:]
    denominator = mc + mp * s**2
    f_val = np.zeros(nx, dtype=Expression)
    f_val[:nq] = qdot * denominator
    f_val[2] = (u + mp * s * (l * qdot[1] ** 2 + g * c))[0]
    f_val[3] = ((-u * c - mp * l * qdot[1] ** 2 * c * s - (mc + mp) * g * s) / l)[0]
    return T @ f_val, denominator


def f2(z, T, dtype=Expression):
    assert len(z) == nz
    s = z[1]
    c = z[2]
    f2_val = np.zeros([nx, nu], dtype=dtype)
    f2_val[2, :] = 1 / (mc + mp * s**2)
    f2_val[3, :] = -c / (mc + mp * s**2) / l
    return T @ f2_val


# State limits (region of state space where we approximate the value function).
d_theta_scale = 1
d_theta = d_theta_scale * np.pi
x_max = np.array([2, np.pi + d_theta, 6, 6])
x_min = np.array([-2, np.pi - d_theta, -6, -6])
u_max = np.array([100])
if d_theta < np.pi / 2:
    z_max = np.array([x_max[0], np.sin(x_min[1]), np.cos(x_min[1]), x_max[2], x_max[3]])
    z_min = np.array([x_min[0], np.sin(x_max[1]), -1, x_min[2], x_min[3]])
else:
    z_max = np.array([x_max[0], 1, np.cos(x_min[1]), x_max[2], x_max[3]])
    z_min = np.array([x_min[0], -1, -1, x_min[2], x_min[3]])
assert (z_min < z_max).all()
x_max_list = list(x_max)
x_max_list[1] = d_theta_scale

d_theta_int = 0.7 * np.pi
x_max_int = np.array([1.5, np.pi + d_theta_int, 4, 4])
x_min_int = np.array([-1.5, np.pi - d_theta_int, -4, -4])
if d_theta_int < np.pi / 2:
    z_max_int = np.array(
        [
            x_max_int[0],
            np.sin(x_min_int[1]),
            np.cos(x_min_int[1]),
            x_max_int[2],
            x_max_int[3],
        ]
    )
    z_min_int = np.array(
        [
            x_min_int[0],
            np.sin(x_max_int[1]),
            -1,
            x_min_int[2],
            x_min_int[3],
        ]
    )
else:
    z_max_int = np.array(
        [x_max_int[0], 1, np.cos(x_min_int[1]), x_max_int[2], x_max_int[3]]
    )
    z_min_int = np.array([x_min_int[0], -1, -1, x_min_int[2], x_min_int[3]])
assert (z_min_int < z_max_int).all()

# Equilibrium point in both the system coordinates.
x0 = np.array([0, np.pi, 0, 0])
z0 = x2z(x0)
z0[np.abs(z0) <= 1e-6] = 0

# Quadratic running cost in augmented state.
Q_diag = [200, 2e3, 2e3, 1e3, 1e3]
Q = np.diag(Q_diag)
R = np.diag([1])


def l_cost(z, u):
    return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)


Rinv = np.linalg.inv(R)


def calc_u_opt(dJdz, f2, Rinv):
    u_star = -0.5 * Rinv.dot(f2.T).dot(dJdz.T)
    return u_star


# x-theta slice of J_star
def plot_value_function(J_star, z):
    dJdz = J_star.ToExpression().Jacobian(z)

    X1, X2 = np.meshgrid(
        np.linspace(x_min[0], x_max[0], 51),
        np.linspace(x_min[1], x_max[1], 51),
    )
    X = np.vstack((X1.flatten(), X2.flatten(), np.zeros(51 * 51), np.zeros(51 * 51)))
    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    U = np.zeros(Z.shape[1])
    RHS = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        z_val = Z[:, i]
        x = X[:, i]
        J[i] = J_star.Evaluate(dict(zip(z, z_val)))
        T_val = T(z_val)
        f2_val = f2(z_val, T_val)
        dJdz_val = np.zeros(nz, dtype=Expression)
        for n in range(nz):
            dJdz_val[n] = dJdz[n].Evaluate(dict(zip(z, z_val)))
        u_opt = calc_u_opt(dJdz_val, f2_val, Rinv)
        U[i] = u_opt

    fig = plt.figure()
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title("Cost-to-Go")
    im = ax.imshow(
        J.reshape(X1.shape),
        cmap=cm.jet,
        aspect="auto",
        extent=(x_min[0], x_max[0], x_max[1], x_min[1]),
    )
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.show()

    fig = plt.figure()
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("theta")
    ax.set_title("Policy")
    im = ax.imshow(
        U.reshape(X1.shape),
        cmap=cm.jet,
        aspect="auto",
        extent=(x_min[0], x_max[0], x_max[1], x_min[1]),
    )
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.show()



def cartpole_sos_lower_bound(deg):
    # Set up optimization.
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    u = prog.NewIndeterminates(nu, "u")
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    # Maximize volume beneath the value function.
    obj = J.Integrate(z[0], z_min_int[0], z_max_int[0])
    for i in range(3, nz):
        obj = obj.Integrate(z[i], z_min_int[i], z_max_int[i])
    cost = 0
    for monomial, coeff in obj.monomial_to_coefficient_map().items():
        s1_deg = monomial.degree(z[1])
        c1_deg = monomial.degree(z[2])
        monomial_int1 = quad(
            lambda x: np.sin(x) ** s1_deg * np.cos(x) ** c1_deg, 0, 2 * np.pi
        )[0]
        if np.abs(monomial_int1) <= 1e-5:
            monomial_int1 = 0
        cost += monomial_int1 * coeff
    poly = Polynomial(cost)
    cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
    # Make the numerics better
    prog.AddLinearCost(-cost / np.max(np.abs(cost_coeff)))

    # Enforce Bellman inequality.
    T_val = T(z)
    f_val, denominator = f(z, u, T_val)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    LHS = J_dot + l_cost(z, u) * denominator

    lam_deg = Polynomial(LHS).TotalDegree()
    # S procedure for s^2 + c^2 = 1.
    lam = prog.NewFreePolynomial(Variables(z), lam_deg).ToExpression()
    S_procedure = lam * (z[1] ** 2 + z[2] ** 2 - 1)
    S_Jdot = 0
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(lam_deg / 2) * 2))[
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

    prog.AddSosConstraint(LHS + S_procedure + S_Jdot)

    # Enforce that value function is PD
    S_J = 0
    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[1] ** 2 + z[2] ** 2 - 1)
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(deg / 2) * 2))[
            0
        ].ToExpression()
        S_J += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])
    # Enforce that value function is PD
    prog.AddSosConstraint(J_expr + S_J + S_r)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Solve and retrieve result.
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    # Only Mosek can solve this example in practice. CSDP is extremely slow, and Clarabel crashes https://github.com/RobotLocomotion/drake/issues/20705.
    mosek_available = MosekSolver().available() and MosekSolver().enabled()
    if not mosek_available:
        print("Mosek is not available. Skipping this example.")
        return Polynomial(Expression(0), z), z
    result = Solve(prog)
    assert result.is_success()
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(
        1e-6
    )

    # Solve for the optimal feedback in augmented coordinates.
    Rinv = np.linalg.inv(R)
    T_val = T(z)
    f2_val = f2(z, T_val)
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)

    return J_star, z

# Note: Lu recommends degree 6, but it takes a few minutes to compute.
J_star, z = cartpole_sos_lower_bound(deg=4)
plot_value_function(J_star, z)


class Controller(LeafSystem):
    def __init__(self, J_star, z, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.x_dim = 4
        self.u_dim = 1
        self.nz = nz
        self.x2z = x2z
        self.T = T
        self.f2 = f2
        self.Rinv = Rinv
        self.dJdz = J_star.Jacobian(z)
        self.z = z

        self.state_input_port = self.DeclareVectorInputPort(
            "state", BasicVector(self.x_dim)
        )
        self.policy_output_port = self.DeclareVectorOutputPort(
            "policy", BasicVector(self.u_dim), self.CalculateController
        )

    def CalculateController(self, context, output):
        x = self.state_input_port.Eval(context)
        z_val = self.x2z(x)
        y = output.get_mutable_value()
        T_val = self.T(z_val)
        f2_val = self.f2(z_val, T_val)
        dJdz_val = np.zeros(self.nz)
        for n in range(self.nz):
            dJdz_val[n] = self.dJdz[n].Evaluate(dict(zip(self.z, z_val)))
        u_opt = calc_u_opt(dJdz_val, f2_val, self.Rinv)
        y[:] = np.clip(u_opt, -180, 180)


def simulate(J_star, z, x0):
    # Animate the resulting policy.
    builder = DiagramBuilder()
    cartpole, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(cartpole)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://underactuated/models/cartpole.urdf")
    cartpole.Finalize()

    wrap = builder.AddSystem(WrapToSystem(4))
    wrap.set_interval(1, 0, 2 * np.pi)
    builder.Connect(cartpole.get_state_output_port(), wrap.get_input_port(0))
    vi_policy = Controller(J_star, z, cartpole)
    builder.AddSystem(vi_policy)
    builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
    builder.Connect(vi_policy.get_output_port(0), cartpole.get_actuation_input_port())

    dt = 0.05
    state_logger = LogVectorOutput(wrap.get_output_port(0), builder, dt)

    meshcat.Delete()
    meshcat.ResetRenderMode()
    viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    context.SetContinuousState(x0)
    viz.StartRecording()
    simulator.AdvanceTo(15)
    viz.StopRecording()
    viz.PublishRecording()


simulate(J_star, z, [0.01, 0, 0, 0])