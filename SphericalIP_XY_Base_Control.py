import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
from utils import calc_u_opt , save_polynomial, load_polynomial
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
    Linearize,
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
from pydrake.examples import PendulumParams
from scipy.integrate import quad
import sympy as sp

from underactuated import ConfigureParser

meshcat = StartMeshcat()

if "MOSEKLM_LICENSE_FILE" not in os.environ:
    # If a mosek.lic file has already been uploaded, then simply use it here.
    if os.path.exists('/home/thomas/mosek/mosek.lic'):
        os.environ["MOSEKLM_LICENSE_FILE"] = "/home/thomas/mosek/mosek.lic"
    else:
        print("stop bein a loooooser and get mosek")

print(MosekSolver().enabled())

def spherical_ip_sos_lower_bound(deg, objective="integrate_ring", visualize=False, test=False, actuator_saturate=False):
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
        f_val[6] = (g / l * cp * st - phi_dot ** 2 * ct * st - u[0] / l * ct + u[
            1] / l * sp * st) * denominator  # thetaddot
        f_val[7] = (g / l * sp + 2 * phi_dot * theta_dot * st - u[1] / l * cp) # phiddot
        return T @ f_val, denominator

    def f2(z, T, dtype=Expression):
        assert len(z) == nz
        st = z[2]
        ct = z[3]
        sp = z[4]
        cp = z[5]
        f2_val = np.zeros([nx, nu], dtype=dtype)
        f2_val[0, :] = [0, 0]
        f2_val[1, :] = [1, 0]
        f2_val[2, :] = [0, 0]
        f2_val[3, :] = [0, 1]
        f2_val[4, :] = [0, 0]
        f2_val[5, :] = [-cp / l, sp * st / l]
        f2_val[6, :] = [0, 0]
        f2_val[7, :] = [0, -cp / (l * ct)]
        return T @ f2_val

    # Define new state limits for the updated system
    d_theta_scale = 0.4
    d_theta = d_theta_scale * np.pi
    d_phi_scale = 0.4
    d_phi = d_phi_scale * np.pi

    x_max = np.array([1.5, 1.5, d_theta, d_phi, 6, 6, 6, 6])
    x_min = np.array([-1.5, -1.5, -d_theta, -d_phi, -6, -6, -6, -6])
    u_max = np.array([30, 30])

    # Compute the z_max and z_min based on the updated system setup
    if d_theta < np.pi / 2 and d_phi < np.pi / 2:
        z_max = np.array(
            [x_max[0], x_max[1], np.sin(x_max[2]), 1, np.sin(x_max[3]), 1, x_max[4], x_max[5],
             x_max[6], x_max[7]])
        z_min = np.array(
            [x_min[0], x_min[1], np.sin(x_min[2]), np.cos(x_min[2]), np.sin(x_min[3]),
             np.cos(x_min[3]), x_min[4], x_min[5], x_min[6], x_min[7]])
    else:
        print("TODO: compute z max for range outside of -pi/2 -> pi/2")
        assert False
        # z_max = np.array(
        #     [x_max[0], x_max[1], 1, np.cos(x_min[2]), 1, np.cos(x_min[3]), x_max[4], x_max[5], x_max[6], x_max[7]])
        # z_min = np.array([x_min[0], x_min[1], -1, -1, -1, -1, x_min[4], x_min[5], x_min[6], x_min[7]])

    # Ensure the transformed state limits are valid
    assert (z_min < z_max).all()

    # Intermediate state limits
    d_theta_int = 0.4 * np.pi
    d_phi_int = 0.4 * np.pi
    x_max_int = np.array([1.5, 1.5, d_theta_int, d_phi_int, 4, 4, 4, 4])
    x_min_int = np.array([-1.5, -1.5, -d_theta_int, -d_phi_int, -4, -4, -4, -4])

    # Compute the z_max and z_min for intermediate limits
    if d_theta_int < np.pi / 2 and d_phi_int < np.pi / 2:
        z_max_int = np.array(
            [x_max_int[0], x_max_int[1], np.sin(x_max_int[2]), 1, np.sin(x_max_int[3]), 1, x_max_int[4], x_max_int[5],
             x_max_int[6], x_max_int[7]])
        z_min_int = np.array(
            [x_min_int[0], x_min_int[1], np.sin(x_min_int[2]), np.cos(x_min_int[2]), np.sin(x_min_int[3]),
             np.cos(x_min_int[3]), x_min_int[4], x_min_int[5], x_min_int[6], x_min_int[7]])
    else:
        print("TODO: compute z max for range outside of -pi/2 -> pi/2")
        assert False
        # z_max_int = np.array([x_max_int[0], x_max_int[1], 1, np.cos(x_min_int[2]), 1, np.cos(x_min_int[3]), x_max_int[4], x_max_int[5], x_max_int[6], x_max_int[7]])
        # z_min_int = np.array([x_min_int[0], x_min_int[1], -1, -1, -1, -1, x_min_int[4], x_min_int[5], x_min_int[6], x_min_int[7]])

    # Ensure the transformed intermediate state limits are valid
    assert (z_min_int < z_max_int).all()

    # Equilibrium point in both the system coordinates.
    # x = (x, y, theta, phi, xdot, ydot, thetadot, phidot)
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    z0 = x2z(x0)
    z0[np.abs(z0) <= 1e-6] = 0

    # Quadratic running cost in augmented state.
    # z = (x, y, st, ct, sp, cp, xdot, ydot, thetadot, phidot)
    # state weighting matrix
    Q_diag = [1e3, 1e3, 11e3, 11e3, 11e3, 11e3, 11e3, 11e3, 11e3, 11e3]#[200, 200, 2000, 2000, 2000, 2000, 1000, 1000, 1000, 1000]
    Q = np.diag(Q_diag)
    # u = (fx fy)
    # control weighting matrix
    R = np.diag([1, 1])

    def l_cost(z, u):
        return (z - z0).dot(Q).dot(z - z0) + u.dot(R).dot(u)
    Rinv = np.linalg.inv(R)

    if test:
        return nz, f, f2, T, x2z, Rinv, z0

    # Set up optimization.
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    u = prog.NewIndeterminates(nu, "u")
    zu = np.concatenate((z, u))
    J = prog.NewFreePolynomial(Variables(z), deg)
    J_expr = J.ToExpression()

    xythetaphi_idx = [0, 1, 6, 7, 8, 9]

    # Maximize volume beneath the value function, integrating over the ring
    # s^2 + c^2 = 1.
    obj = J
    for i in xythetaphi_idx:
        obj = obj.Integrate(z[i], z_min_int[i], z_max_int[i])
    cost = 0
    for monomial, coeff in obj.monomial_to_coefficient_map().items():
        s1_deg = monomial.degree(z[2])  # sin(theta)
        c1_deg = monomial.degree(z[3])  # cos(theta)
        s2_deg = monomial.degree(z[4])  # sin(phi)
        c2_deg = monomial.degree(z[5])  # cos(phi)
        monomial_int = quad(lambda x: np.sin(x) ** s1_deg * np.cos(x) ** c1_deg * np.sin(x) ** s2_deg * np.cos(x) ** c2_deg, 0, 2 * np.pi)[0]
        if np.abs(monomial_int) <= 1e-5:
            monomial_int1 = 0
        cost += monomial_int*coeff

    poly = Polynomial(cost)
    cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
    # Make the numerics better
    prog.AddLinearCost(-cost / np.max(np.abs(cost_coeff)))

    # Enforce Lyapunov function is negative definite (bellman inequality for optimal)
    T_val = T(z)
    f_val, denominator = f(z, u, T_val)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    LHS = J_dot + l_cost(z, u) * denominator  # Lower bound on cost to go V >= -l  -->  V + l >= 0
    # LHS = J_dot + 1*denominator# Relaxed Hamilton jacobian bellman conditions, non optimal, but still lyapunov

    lam_deg = Polynomial(LHS).TotalDegree() - 2
    lam_deg = 2

    # S procedure for st^2 + ct^2 + sp^2 + cp^2 = 2.
    lam = prog.NewFreePolynomial(Variables(zu), lam_deg).ToExpression()
    S_procedure = lam * (z[2] ** 2 + z[3] ** 2 * z[4] ** 2 + z[5] ** 2 * z[3] ** 2 - 1)
    S_Jdot = 0
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(zu), lam_deg)[0].ToExpression()  # doesnt have to be SOS!! bc we're dealing with "g"==0 not <=0
        S_Jdot += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])  # negative inside the range of z-space we are defining to be locally stable

    # Enforce Input constraint
    u_min = -u_max
    if actuator_saturate:
        for i in range(nu):
            lam = prog.NewSosPolynomial(Variables(zu), lam_deg)[0].ToExpression()
            S_Jdot += lam * (u[i] - u_max[i]) * (u[i] - u_min[i])
    prog.AddSosConstraint(LHS + S_procedure + S_Jdot)

    # Enforce that value function is Positive Definite
    S_J = 0
    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[2] ** 2 + z[3] ** 2 * z[4] ** 2 + z[5] ** 2 * z[3] ** 2 - 1)  # S-procedure again
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), deg)[0].ToExpression()
        S_J += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])  # +ve when z > zmax or z < zmin, -ve inside z bounds
    prog.AddSosConstraint(J_expr + S_J + S_r)


    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # J(z_max(or min)) = mgl NOT WORKING AS IS
    J_max = J_expr.EvaluatePartial(dict(zip(z, z_max_int)))
    # V.Substitute({st: 1, ct: 0, thetadot: 0, sp: 1, cp: 0, psidot: 0}) == m2 * g * Len)
    J_max = J_expr.Substitute({z[0]: 0, z[1]: 0, z[2]: 1, z[3]: 0, z[4]: 1, z[5]: 0, z[6]: 0, z[7]: 0, z[8]: 0, z[9]: 0})
    # prog.AddLinearConstraint(J_max == mp*g*l)

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
    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-9)
    os.makedirs("SphericalIP/data/{}".format(z_max), exist_ok=True)
    save_polynomial(J_star, z, "SphericalIP/data/{}/J_lower_bound_deg_{}.pkl".format(z_max, deg))

    # Solve for the optimal feedback in augmented coordinates.
    Rinv = np.linalg.inv(R)
    T_val = T(z)
    f2_val = f2(z, T_val)
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)
    print(f"Control Law: {u_star}")

    # Define symbolic variables
    x, y, theta, phi, x_dot, y_dot, theta_dot, phi_dot = sp.symbols('x y theta phi x_dot y_dot theta_dot phi_dot')
    s_theta = sp.sin(theta)
    c_theta = sp.cos(theta)
    s_phi = sp.sin(phi)
    c_phi = sp.cos(phi)

    # Parse the string to a SymPy expression
    fx = sp.sympify(u_star[0])
    fy = sp.sympify(u_star[1])
    # Map z matrix parameters to symbolic representations
    fx_str_sub = fx.__str__().replace("z(0)", "x") \
        .replace("z(1)", "y") \
        .replace("z(2)", "sin(theta)") \
        .replace("z(3)", "cos(theta)") \
        .replace("z(4)", "sin(phi)") \
        .replace("z(5)", "cos(phi)") \
        .replace("z(6)", "x_dot") \
        .replace("z(7)", "y_dot") \
        .replace("z(8)", "theta_dot") \
        .replace("z(9)", "phi_dot")
    fy_str_sub = fy.__str__().replace("z(0)", "x") \
        .replace("z(1)", "y") \
        .replace("z(2)", "sin(theta)") \
        .replace("z(3)", "cos(theta)") \
        .replace("z(4)", "sin(phi)") \
        .replace("z(5)", "cos(phi)") \
        .replace("z(6)", "x_dot") \
        .replace("z(7)", "y_dot") \
        .replace("z(8)", "theta_dot") \
        .replace("z(9)", "phi_dot")
    print(f"fx: {fx_str_sub}")
    print(f"fy: {fy_str_sub}")

    if visualize:
        plot_value_function(J_star, z, z_max, plot_states="xtheta", u_index=0)

    return J_star, z

def plot_value_function(J_star, z, z_max, plot_states="xy", u_index=0):
    nz = 10
    x_max = np.zeros(8)
    x_max[:2] = z_max[:2]
    x_max[2] = np.pi/2
    x_max[3] = np.pi / 2
    x_max[4:] = z_max[6:]
    x_min = -x_max

    dJdz = J_star.ToExpression().Jacobian(z)
    nz, f, f2, T, x2z, Rinv, z0 = spherical_ip_sos_lower_bound(2, test=True)
    zero_vector = np.zeros(51*51)
    if plot_states == "xtheta":
        X1, THETA = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[2], x_max[2], 51))
        X = np.vstack((X1.flatten(), zero_vector, THETA.flatten(), zero_vector, zero_vector, zero_vector, zero_vector, zero_vector))
        ylabel="theta"
    elif plot_states == "xy":
        X1, Y = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
        X = np.vstack((X1.flatten(), Y.flatten(), zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, zero_vector))
        ylabel="y"

    Z = x2z(X)
    J = np.zeros(Z.shape[1])
    U = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        z_val = Z[:, i]
        T_val = T(z_val)
        J[i] = J_star.Evaluate(dict(zip(z, z_val)))
        f2_val = f2(z_val, T_val, dtype=float)
        dJdz_val = np.zeros(nz)
        for n in range(nz):
            dJdz_val[n] = dJdz[n].Evaluate(dict(zip(z, z_val)))
        U[i] = calc_u_opt(dJdz_val, f2_val, Rinv)[u_index]

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title("Cost-to-Go")
    im = ax.imshow(J.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[2], x_min[2]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.show()

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title("Policy")
    im = ax.imshow(U.reshape(X1.shape),
            cmap=cm.jet, aspect='auto',
            extent=(x_min[0], x_max[0], x_max[2], x_min[2]))
    ax.invert_yaxis()
    fig.colorbar(im)
    plt.show()

def lqr(Q, nz, nu, mp, l):
    g = 9.81

    A = np.zeros([nz, nz])
    B = np.zeros([nz, nu])
    A[0, 3] = 1
    A[1, -1] = -1
    A[3, 1] = -mp*g
    A[4, 1] = -g*(mp)
    B[3, :] = 1
    B[4, :] = 1
    F = np.array([0, 0, -1, 0, 0])
    R = np.diag([1])
    K, S = LinearQuadraticRegulator(A, B, Q, R, F=F.reshape(1, nz))
    return np.squeeze(K), S


J_star, z = spherical_ip_sos_lower_bound(2, visualize=True, actuator_saturate=False)
# print(J_star)

