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

def sk8dyn_sos_lower_bound(deg, objective="integrate_ring", constraint = "kSos", visualize=False, test=False, actuator_saturate=False, read_file=None):
    nz = 8
    nq = 3
    nx = 2 * nq
    nu = 2

    # Kinematic Parameters
    m1 = .4# mass of board
    m2 = 9.0  # mass of rider
    L = 0.9  # half height of rider
    l = 1  # length of board
    g = 9.81  # gravity, duh
    Ib = 0.475  # moment of inertia of board+feet
    Kphi = 250  # skateboard truck spring constant

    # Human control parameters
    Kpy = 1066
    Kdy = 349
    Wprop = 0.5
    Ky = 94
    Cy = 5.8
    Kpprop = 1200
    Kpv = 1000
    Kdprop = 210
    Kdv = 500

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (x, theta, psi, xdot, thetadot, psidot)
    # z = (x, st, ct, sp, cp, xdot, thetadot, psidot)
    x2z = lambda x: np.array([x[0], np.sin(x[1]), np.cos(x[1]), np.sin(x[2]), np.cos(x[2]), x[3], x[4], x[5]])

    # Transform to augmented representation
    def T(z, dtype=Expression):
        assert len(z) == nz
        T = np.zeros([nz, nx], dtype=dtype)
        T[0, 0] = 1
        T[1, 1] = z[2]
        T[2, 1] = -z[1]
        T[3, 2] = z[4]
        T[4, 2] = -z[3]
        T[5, 3] = 1
        T[6, 4] = 1
        T[7, 5] = 1
        return T

    # z = (x, st, ct, sp, cp, xdot, thetadot, psidot)
    def f(z, u, T, dtype=Expression):
        st = z[1]
        ct = z[2]
        sp = z[3]
        cp = z[4]
        x_dot = z[5]
        theta_dot = z[6]
        psi_dot = z[7]

        assert len(z) == nz
        assert len(u) == nu
        if dtype == float:
            assert (st ** 2 + ct ** 2 * sp ** 2 + cp ** 2 * ct ** 2) == 1

        qdot = z[-nq:]
        # denominator multipliers
        denom1 = ct**2
        denom2 = m1+m2*st**2
        denominator = denom1*denom2
        f_val = np.zeros(nx, dtype=Expression)
        f_val[:nq] = qdot * denominator
        f_val[3] = (m2 * L * st * (theta_dot ** 2 + psi_dot ** 2 * ct ** 2) - g * m2 * st * ct * cp + u[0] + u[1]*sp*st*ct/L) * denom1  # Denom is m1+m2*st**2  # xddot
        f_val[4] = (g*m2*cp*st/L -m1*psi_dot**2*st*ct + m1*g*cp*st/L - (psi_dot**2+theta_dot**2)*m2*st*ct \
                     - u[0]*ct/L - (m1+m2)*u[1]*sp*st/(m2*L**2)) * denom1  # Denom is m1+m2*st**2  # thetaddot
        f_val[5] = ((2*psi_dot**2*theta_dot**2*st + g*sp/L)*ct + u[1]*cp/(m2*L**2)) * denom2  # Denom is ct**2 # psiddot

        return T @ f_val, denominator

    def f2(z, T, dtype=Expression):
        st = z[1]
        ct = z[2]
        sp = z[3]
        cp = z[4]
        x_dot = z[5]
        theta_dot = z[6]
        psi_dot = z[7]

        assert len(z) == nz
        denom1 = ct ** 2
        denom2 = m1 + m2 * st ** 2

        f2_val = np.zeros([nx, nu], dtype=dtype)
        f2_val[3, :] = [1/denom2, sp*st*ct/(L*denom2)]  # xddot
        f2_val[4, :] = [-ct/(L*denom2), -(m1+m2)*sp*st/(m2*L**2*denom2)]  # thetaddot
        f2_val[5, :] = [0, cp/(m2*L**2*denom1)]  # psiddot

        return T @ f2_val


    # Define State Limits
    # x = (x, theta, psi, xdot, thetadot, psidot)
    # z = (x, st, ct, sp, cp, xdot, thetadot, psidot)
    u_max = np.array([30, 30])
    z_max = np.array([1, np.sin(np.pi/2), 1, np.sin(np.pi / 2), 1, 1, 1, 1])
    z_min = np.array([-1, -np.sin(np.pi/2), 0, -np.sin(np.pi/2), 0, -1, -1, -1])
    assert (z_min < z_max).all()

    # Equilibrium point in both the system coordinates.
    x0 = np.zeros(nx)
    z0 = x2z(x0)
    z0[np.abs(z0) <= 1e-6] = 0

    # Quadratic running cost in augmented state.
    # state weighting matrix
    # z = (x, st, ct, sp, cp, xdot, thetadot, psidot)
    Q_diag = [0.0, 20000, 20000, 20000, 20000, 0.0, 1, 1]

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

    if read_file:
        filename = read_file
        J_star = load_polynomial(z, filename+".pkl")
        Rinv = np.linalg.inv(R)
        T_val = T(z)
        f2_val = f2(z, T_val)
        dJdz = J_star.Jacobian(z)
        u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)
        uToStr(u_star, filename+".txt")
        f_val, denom = f(z, u, T_val)
        # degF = max([Polynomial(i).TotalDegree() for i in f_val])
        # deg_Ustar = max([[Polynomial(i, z).TotalDegree() for i in j] for j in u_star])
        plot_value_function(J_star, z, z_max, u_max, plot_states="thetapsi", actuator_saturate=False)
        return 0


    # Configure Optimization Strategy
    if constraint == "kSos":
        constraint_type = prog.NonnegativePolynomial.kSos
    elif constraint == "kSdsos":
        constraint_type = prog.NonnegativePolynomial.kSdsos
    else:
        constraint_type = prog.NonnegativePolynomial.kDsos

    xphithetapsi_idx = [0, 5, 6, 7]

    # Maximize volume beneath the value function, integrating over the ring
    # s^2 + c^2 = 1.
    obj = J
    for i in xphithetapsi_idx:
        obj = obj.Integrate(z[i], z_min[i], z_max[i])
    cost = 0
    for monomial, coeff in obj.monomial_to_coefficient_map().items():
        s2_deg = monomial.degree(z[1])  # sin(theta)
        c2_deg = monomial.degree(z[2])  # cos(theta)
        s3_deg = monomial.degree(z[3])  # sin(psi)
        c3_deg = monomial.degree(z[4])  # cos(psi)
        monomial_int = quad(lambda x: np.sin(x) ** s2_deg * np.cos(x) ** c2_deg * np.sin(x) ** s3_deg * np.cos(x) ** c3_deg, -np.pi/2, np.pi/2)[0]
        if np.abs(monomial_int) <= 1e-5:
            monomial_int1 = 0
        cost += monomial_int*coeff
    poly = Polynomial(cost)
    cost_coeff = [c.Evaluate() for c in poly.monomial_to_coefficient_map().values()]
    # Make the numerics better
    prog.AddLinearCost(-cost / np.max(np.abs(cost_coeff)))

    # Enforce Lyapunov function is negative definite (HJB inequality)
    T_val = T(z)
    f_val, denominator = f(z, u, T_val)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    LHS = J_dot + l_cost(z, u) * denominator  # Lower bound on cost to go V >= -l  -->  V + l >= 0
    # LHS = J_dot + 1*denominator  # Relaxed Hamilton jacobian bellman conditions, non-optimal, but still lyapunov

    ring_deg = deg
    # z = (x, st, ct, sp, cp, xdot, thetadot, psidot)
    # S procedure for rigid body
    lam = prog.NewFreePolynomial(Variables(zu), ring_deg).ToExpression()  # doesnt have to be SOS!! bc we're dealing with "g"==0 not <=0
    S_sphere = lam * (z[1] ** 2 + z[2] ** 2 * z[3] ** 2 + z[4] ** 2 * z[2] ** 2 - 1)
    # S procedure for z_min < z < z_max
    S_Jdot = 0
    for i in np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(zu), ring_deg, type=constraint_type)[0].ToExpression()
        S_Jdot += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])  # negative inside the range of z-space we are defining to be locally stable

    # Enforce Input constraint
    u_min = -u_max
    if actuator_saturate:
        for i in range(nu):
            lam = prog.NewSosPolynomial(Variables(zu), ring_deg, type=constraint_type)[0].ToExpression()
            S_Jdot += lam * (u[i] - u_max[i]) * (u[i] - u_min[i])
    prog.AddSosConstraint(LHS + S_sphere*denominator + S_Jdot*denominator, type=constraint_type)

    # Enforce that value function is Positive Definite
    S_J = 0
    lam_r = prog.NewFreePolynomial(Variables(z), deg).ToExpression()
    S_r = lam_r * (z[1] ** 2 + z[2] ** 2 * z[3] ** 2 + z[4] ** 2 * z[2] ** 2 - 1)  # S-procedure again
    # S procedure for z_min < z < z_max
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
    os.makedirs("Sk8Dyn/data/spherical/{}/{}".format(z_max, constraint), exist_ok=True)
    save_polynomial(J_star, z, "Sk8Dyn/data/spherical/{}/{}/J_lower_bound_deg_{}.pkl".format(z_max, constraint, deg))
    uToStr(u_star, "Sk8Dyn/data/spherical/{}/{}/J_lower_bound_deg_{}.txt".format(z_max, constraint, deg))

    if visualize:
        plot_value_function(J_star, z, z_max, u_max, plot_states="thetapsi", actuator_saturate=actuator_saturate)

    return J_star, z

def plot_value_function(J_star, z, z_max, u_max, plot_states="xy", actuator_saturate=False):
    nz = 8
    x_max = np.zeros(6)
    x_max[0] = z_max[0]
    x_max[1] = np.pi/2-.1
    x_max[2] = np.pi/2-.1
    x_max[3:] = z_max[5:]
    x_min = -x_max

    try:
        dJdz = J_star.ToExpression().Jacobian(z)
    except:
        dJdz = J_star.Jacobian(z)

    zero_vector = np.zeros(51*51)
    xdot_vector = np.ones(51*51)*10
    # x = (x, theta, psi, xdot, thetadot, psidot)
    # z = (x, st, ct, sp, cp, xdot, thetadot, psidot)
    if plot_states == "xtheta":
        X1, THETA = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[1], x_max[1], 51))
        X = np.vstack((X1.flatten(), THETA.flatten(), zero_vector, xdot_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, X, X1, x_min, x_max, z_max, u_max, 0, 1, "x", "theta", actuator_saturate=actuator_saturate)
    elif plot_states == "thetapsi":
        THETA1, PSI = np.meshgrid(np.linspace(x_min[1], x_max[1], 51),
                        np.linspace(x_min[2], x_max[2], 51))
        THETA = np.vstack((zero_vector, THETA1.flatten(), PSI.flatten(), xdot_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, THETA, THETA1, x_min, x_max, z_max, u_max, 1, 2, "theta", "psi", actuator_saturate=actuator_saturate)
    elif plot_states == "thetadot":
        THETA1, THETADOT = np.meshgrid(np.linspace(x_min[1], x_max[1], 51),
                        np.linspace(x_min[4], x_max[4], 51))
        THETA = np.vstack((zero_vector, THETA1.flatten(), zero_vector, xdot_vector, THETADOT.flatten(), zero_vector))
        plotCostPolicy(dJdz, J_star, z, THETA, THETA1, x_min, x_max, z_max, u_max, 1, 4, "theta", "theta_dot", actuator_saturate=actuator_saturate)
    elif plot_states == "psidot":
        PSI1, PSIDOT = np.meshgrid(np.linspace(x_min[2], x_max[2], 51),
                        np.linspace(x_min[5], x_max[5], 51))
        PSI = np.vstack((zero_vector, zero_vector, PSI1.flatten(), xdot_vector, zero_vector, PSIDOT.flatten()))
        plotCostPolicy(dJdz, J_star, z, PSI, PSI1, x_min, x_max, z_max, u_max, 2, 5, "psi", "psi_dot", actuator_saturate=actuator_saturate)



def plotCostPolicy(dJdz, J_star, z, X, X1, x_min, x_max, z_max, u_max, xaxis_ind, yaxis_ind, xlabel, ylabel, actuator_saturate=False):
    nz, f, f2, T, x2z, Rinv, z0 = sk8dyn_sos_lower_bound(2, test=True)
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
                U[_] = np.clip(U[_], -1000, 1000)

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
        .replace("z(1)", "sin(theta)") \
        .replace("z(2)", "cos(theta)") \
        .replace("z(3)", "sin(psi)") \
        .replace("z(4)", "cos(psi)") \
        .replace("z(5)", "x_dot") \
        .replace("z(6)", "theta_dot") \
        .replace("z(7)", "psi_dot")
    fy_str_sub = fy.__str__().replace("z(0)", "x") \
        .replace("z(1)", "sin(theta)") \
        .replace("z(2)", "cos(theta)") \
        .replace("z(3)", "sin(psi)") \
        .replace("z(4)", "cos(psi)") \
        .replace("z(5)", "x_dot") \
        .replace("z(6)", "theta_dot") \
        .replace("z(7)", "psi_dot")
    fx_str_sub= re.sub(r"pow\(([^,]+),\s*(\d+)\)", r"\1^\2", fx_str_sub)
    fy_str_sub = re.sub(r"pow\(([^,]+),\s*(\d+)\)", r"\1^\2", fy_str_sub)
    print(f"fx: {fx_str_sub}")
    print(f"fy: {fy_str_sub}")

    # Save to text file for Transfer to Matlab
    if file:
        with open(file, "w") as text_file:
            text_file.write(fx_str_sub+"\n"+fy_str_sub)


# filename = "/home/thomas/Documents/thesis/LyapunovConvexOptimization/SphericalIP/data/[1.5 1.5 1.  1.  1.  1.  4.  4.  3.  3. ]/J_lower_bound_deg_4_SOS_Q20000"
filename = "/home/thomas/Documents/thesis/LyapunovConvexOptimization/Sk8Dyn/data/spherical/[1.5 1.  1.  1.  1.  5.  4.  4. ]/kSdsos/J_lower_bound_deg_6"
J_star, z = sk8dyn_sos_lower_bound(6, constraint = "kSdsos", visualize=True, actuator_saturate=False, read_file=False)

# TRYIN TO SEE IF DEG6 CAN SEE THE PROPER DYNAMICS WITH 1/10Mass as deg4 didnt see it