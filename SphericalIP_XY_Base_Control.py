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

def spherical_ip_sos_lower_bound(deg, objective="integrate_ring", constraint = "kSos", visualize=False, test=False, actuator_saturate=False, plot_saved=False):
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


    # Define State Limits
    u_max = np.array([30, 30])
    z_max = np.array([1.5, 1.5, np.sin(np.pi/2), 1, np.sin(np.pi / 2), 1, 4, 4, 3, 3])
    z_min = np.array([-1.5, -1.5, -np.sin(np.pi/2), 0, -np.sin(np.pi/2), 0, -4, -4, -3, -3])
    assert (z_min < z_max).all()

    # Equilibrium point in both the system coordinates.
    # x = (x, y, theta, phi, xdot, ydot, thetadot, phidot)
    x0 = np.zeros(nx)
    z0 = x2z(x0)
    z0[np.abs(z0) <= 1e-6] = 0

    # Quadratic running cost in augmented state.
    # z = (x, y, st, ct, sp, cp, xdot, ydot, thetadot, phidot)
    # state weighting matrix
    Q_diag = [0.01, 0.01, 20000, 20000, 20000, 20000, 1, 1, 1, 1]
    Q = np.diag(Q_diag)
    # u = (fx fy)
    # control weighting matrix
    R = np.array([[0.1, 0.0], [0.0, 0.1]])

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

    if plot_saved:
        filename = "SphericalIP/data/old_0_2pi_int_bounds/J_lower_bound_deg_4_SDSOS_Q20000"
        J_star = load_polynomial(z, filename+".pkl")
        Rinv = np.linalg.inv(R)
        T_val = T(z)
        f2_val = f2(z, T_val)
        dJdz = J_star.Jacobian(z)
        u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)
        uToStr(u_star, filename+".txt")
        plot_value_function(J_star, z, z_max, u_max, plot_states="thetaphi", actuator_saturate=False)
        return 0


    # Configure Optimization Strategy
    if constraint == "kSos":
        constraint_type = prog.NonnegativePolynomial.kSos
    elif constraint == "kSdsos":
        constraint_type = prog.NonnegativePolynomial.kSdsos
    else:
        constraint_type = prog.NonnegativePolynomial.kDsos

    xythetaphi_idx = [0, 1, 6, 7, 8, 9]

    # Maximize volume beneath the value function, integrating over the ring
    # s^2 + c^2 = 1.
    obj = J
    for i in xythetaphi_idx:
        obj = obj.Integrate(z[i], z_min[i], z_max[i])
    cost = 0
    for monomial, coeff in obj.monomial_to_coefficient_map().items():
        s1_deg = monomial.degree(z[2])  # sin(theta)
        c1_deg = monomial.degree(z[3])  # cos(theta)
        s2_deg = monomial.degree(z[4])  # sin(phi)
        c2_deg = monomial.degree(z[5])  # cos(phi)
        monomial_int = quad(lambda x: np.sin(x) ** s1_deg * np.cos(x) ** c1_deg * np.sin(x) ** s2_deg * np.cos(x) ** c2_deg, -np.pi/2, np.pi/2)[0]
        # monomial_int = quad(lambda x: np.sin(x) ** s2_deg * np.cos(x) ** c2_deg, -np.pi/2, np.pi/2)[0]
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
    # LHS = J_dot + 10*denominator# Relaxed Hamilton jacobian bellman conditions, non-optimal, but still lyapunov

    ring_deg = 4
    # S procedure for st^2 + ct^2 + sp^2 + cp^2 = 2.
    lam = prog.NewFreePolynomial(Variables(zu), ring_deg).ToExpression()
    S_sphere = lam * (z[2] ** 2 + z[3] ** 2 * z[4] ** 2 + z[5] ** 2 * z[3] ** 2 - 1)
    S_Jdot = 0
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(zu), ring_deg, type=constraint_type)[0].ToExpression()  # doesnt have to be SOS!! bc we're dealing with "g"==0 not <=0
        S_Jdot += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])  # negative inside the range of z-space we are defining to be locally stable

    # Enforce Input constraint
    u_min = -u_max
    if actuator_saturate:
        for i in range(nu):
            lam = prog.NewSosPolynomial(Variables(zu), ring_deg, type=constraint_type)[0].ToExpression()
            S_Jdot += lam * (u[i] - u_max[i]) * (u[i] - u_min[i])
    prog.AddSosConstraint(LHS + S_sphere + S_Jdot, type=constraint_type)

    # Enforce that value function is Positive Definite
    S_J = 0
    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[2] ** 2 + z[3] ** 2 * z[4] ** 2 + z[5] ** 2 * z[3] ** 2 - 1)  # S-procedure again
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), deg, type=constraint_type)[0].ToExpression()
        S_J += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])  # +ve when z > zmax or z < zmin, -ve inside z bounds
    prog.AddSosConstraint(J_expr + S_J + S_r, type=constraint_type)

    # J(z0) = 0.
    J0 = J_expr.EvaluatePartial(dict(zip(z, z0)))
    prog.AddLinearConstraint(J0 == 0)

    # Solve and retrieve result.
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    mosek_available = MosekSolver().available() and MosekSolver().enabled()
    if not mosek_available:
        print("Mosek is not available. Skipping this example.")
        return Polynomial(Expression(0), z), z
    result = Solve(prog)
    # assert result.is_success()

    J_star = Polynomial(result.GetSolution(J_expr)).RemoveTermsWithSmallCoefficients(1e-6)

    # Solve for the optimal feedback in augmented coordinates.
    Rinv = np.linalg.inv(R)
    T_val = T(z)
    f2_val = f2(z, T_val)
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)

    # Save data to file
    os.makedirs("SphericalIP/data/{}/{}".format(z_max, constraint), exist_ok=True)
    save_polynomial(J_star, z, "SphericalIP/data/{}/{}/J_lower_bound_deg_{}.pkl".format(z_max, constraint, deg))
    uToStr(u_star, "SphericalIP/data/{}/{}/J_lower_bound_deg_{}.txt".format(z_max, constraint, deg))

    if visualize:
        plot_value_function(J_star, z, z_max, u_max, plot_states="thetaphi", actuator_saturate=actuator_saturate)

    return J_star, z

def plot_value_function(J_star, z, z_max, u_max, plot_states="xy", actuator_saturate=False):
    nz = 10
    x_max = np.zeros(8)
    x_max[:2] = z_max[:2]
    x_max[2] = np.pi/2
    x_max[3] = np.pi / 2
    x_max[4:] = z_max[6:]
    x_min = -x_max

    try:
        dJdz = J_star.ToExpression().Jacobian(z)
    except:
        dJdz = J_star.Jacobian(z)

    zero_vector = np.zeros(51*51)
    if plot_states == "xtheta":
        X1, THETA = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[2], x_max[2], 51))
        X = np.vstack((X1.flatten(), zero_vector, THETA.flatten(), zero_vector, zero_vector, zero_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, X, X1, x_min, x_max, z_max, u_max, 0, 2, "x", "theta", actuator_saturate=actuator_saturate)
    elif plot_states == "yphi":
        Y1, PHI = np.meshgrid(np.linspace(x_min[1], x_max[1], 51),
                        np.linspace(x_min[3], x_max[3], 51))
        Y = np.vstack(( zero_vector, Y1.flatten(), zero_vector, PHI.flatten(), zero_vector, zero_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, Y, Y1, x_min, x_max, z_max, u_max,1, 3, "y", "phi", actuator_saturate=actuator_saturate)
    elif plot_states == "thetaphi":
        THETA1, PHI = np.meshgrid(np.linspace(x_min[2], x_max[2], 51),
                        np.linspace(x_min[3], x_max[3], 51))
        THETA = np.vstack((zero_vector, zero_vector, THETA1.flatten(), PHI.flatten(), zero_vector, zero_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, THETA, THETA1, x_min, x_max, z_max, u_max, 2, 3, "theta", "phi", actuator_saturate=actuator_saturate)
    elif plot_states == "xy":
        X1, Y = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
        X = np.vstack((X1.flatten(), Y.flatten(), zero_vector, zero_vector, zero_vector, zero_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, X, X1, x_min, x_max, z_max, u_max, 0, 1, "x", "y", actuator_saturate=actuator_saturate)




def plotCostPolicy(dJdz, J_star, z, X, X1, x_min, x_max, z_max, u_max, xaxis_ind, yaxis_ind, xlabel, ylabel, actuator_saturate=False):
    nz, f, f2, T, x2z, Rinv, z0 = spherical_ip_sos_lower_bound(2, test=True)
    Z = x2z(X)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # figsize can be adjusted as needed
    for i in range(2):

        J = np.zeros(Z.shape[1])
        U = np.zeros(Z.shape[1])
        for _ in range(Z.shape[1]):
            z_val = Z[:, _]
            T_val = T(z_val)
            J[_] = J_star.Evaluate(dict(zip(z, z_val)))
            f2_val = f2(z_val, T_val, dtype=float)
            dJdz_val = np.zeros(nz)
            for n in range(nz):
                dJdz_val[n] = dJdz[n].Evaluate(dict(zip(z, z_val)))
            U[_] = calc_u_opt(dJdz_val, f2_val, Rinv)[i]
            if actuator_saturate:
                U[_] = np.clip(U[_], -u_max[i], u_max[i])
            else:
                U[_] = np.clip(U[_], -60, 60)

        axs[i,0].set_xlabel(xlabel)
        axs[i,0].set_ylabel(ylabel)
        axs[i,0].set_title(f"Cost-to-Go, Input #: {i}")
        im = axs[i,0].imshow(J.reshape(X1.shape),
                       cmap=cm.jet, aspect='auto',
                       extent=(x_min[xaxis_ind], x_max[xaxis_ind], x_max[yaxis_ind], x_min[yaxis_ind]))
        axs[i,0].invert_yaxis()
        fig.colorbar(im, ax=axs[i,0])

        axs[i,1].set_xlabel(xlabel)
        axs[i,1].set_ylabel(ylabel)
        axs[i,1].set_title(f"Policy, Input #: {i}")
        im = axs[i,1].imshow(U.reshape(X1.shape),
                       cmap=cm.jet, aspect='auto',
                       extent=(x_min[xaxis_ind], x_max[xaxis_ind], x_max[yaxis_ind], x_min[yaxis_ind]))
        axs[i,1].invert_yaxis()
        fig.colorbar(im, ax=axs[i,1])

    plt.tight_layout()
    plt.show()

def uToStr(U, file=None):
    # Define symbolic variables
    # Parse the string to a SymPy expression
    fx = U[0]
    fy = U[1]
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
    fx_str_sub= re.sub(r"pow\(([^,]+),\s*(\d+)\)", r"\1^\2", fx_str_sub)
    fy_str_sub = re.sub(r"pow\(([^,]+),\s*(\d+)\)", r"\1^\2", fy_str_sub)
    print(f"fx: {fx_str_sub}")
    print(f"fy: {fy_str_sub}")

    # Save to text file for Transfer to Matlab
    if file:
        with open(file, "w") as text_file:
            text_file.write(fx_str_sub+"\n"+fy_str_sub)


    def roa():
        sys = SymbolicVectorSystem(state=[x], dynamics=[-x + x**3])
        context = sys.CreateDefaultContext()
        V = RegionOfAttraction(system=sys, context=context)

        print("Verified that " + str(V) + " < 1 is in the region of attraction.")




J_star, z = spherical_ip_sos_lower_bound(4, constraint = "kSdsos", visualize=True, actuator_saturate=True, plot_saved=False)
