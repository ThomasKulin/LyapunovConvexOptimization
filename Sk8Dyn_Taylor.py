import matplotlib
import pydrake.all

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
    TaylorExpand,
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
from pathlib import Path
import dill

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
    nq = 4
    nx = 2 * nq
    nu = 2

    # Kinematic Parameters
    m1 = 4  # mass of board
    m2 = 90  # mass of rider
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
    # x = (x, phi, theta, psi, xdot, phidot, thetadot, psidot)
    x2z = lambda x: np.array([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]])  # z is same as x, too lazy to rework

    # z = (x, st, ct, sp, cp, xdot, thetadot, psidot)
    def f(z, u, dtype=Expression):
        assert len(z) == nz
        assert len(u) == nu

        phi = z[1]
        sph = np.sin(z[1])
        cph = np.cos(z[1])
        theta = z[2]
        st = np.sin(z[2])
        ct = np.cos(z[2])
        psi = z[3]
        sps = np.sin(z[3])
        cps = np.cos(z[3])
        x_dot = z[4]
        phi_dot = z[5]
        theta_dot = z[6]
        psi_dot = z[7]

        # Ankle torque inputs
        tau_t = 0#(-(Kpprop + Kpv) * theta - (Kdprop + Kdv) * theta_dot)
        tau_ps = 0#(-Kpy * (psi - Wprop * phi) - Kdy * (psi_dot - Wprop * phi_dot) - Ky * (psi - phi) - Cy * (psi_dot - phi_dot))
        tau_ph = (-tau_ps)

        # Turning radius - y-position calculation (not including diff torque input!). r vector for centrifugal force calc
        ry_num = l * cph# - 2 * L * ct * sps * sph #/ (2 * sph) MULTIPLYING EVERYTHING BY THE DENOM TO AVOID SINGULARITY
        ry_den = 2 * sph
        qdot = z[-nq:]
        # denominator multipliers
        denom1 = ct ** 2
        denom2 = m1 + m2 * st ** 2
        f_val = np.zeros(nx, dtype=Expression)
        f_val[:nq] = qdot
        f_val[4] = (m2 * L * st * (theta_dot ** 2 + psi_dot ** 2 * ct ** 2) - g * m2 * st * ct * cps + u[0] + u[
            1] * sps * st * ct / L - tau_t * ct / L) / denom2  # xddot
        f_val[5] = (tau_ph - Kphi * phi) / Ib  # phiddot
        f_val[6] = (g * m2 * cps * st / L - m1 * psi_dot ** 2 * st * ct + m1 * g * cps * st / L - (
                psi_dot ** 2 + theta_dot ** 2) * m2 * st * ct \
                    - u[0] * ct / L + (m1 + m2) * (
                            tau_t - (m2 * x_dot ** 2 * ry_den / ry_num + u[1]) * sps * st) / (
                            m2 * L ** 2)) / denom2  # thetaddot
        f_val[7] = ((2 * psi_dot * theta_dot * st + g * sps / L) / ct + (
                tau_ps + (m2 * x_dot ** 2 * ry_den / ry_num + u[1]) * cps) / (m2 * L ** 2 * ct ** 2))  # psiddot

        order = 5
        filepath = Path(f'Sk8Dyn/data/Taylor/Dynamics_TaylorDegree{order}.npy')
        if filepath.exists():
            f_sym = np.load(filepath, allow_pickle=True)
            varlist, _ = pydrake.all.ExtractVariablesFromExpression(f_val)
            memo = dict()
            _ = [pydrake.all.to_sympy(var, memo=memo) for var in varlist]  # recover memo dictionary to recreate expression
            f_val = [pydrake.all.from_sympy(f, memo=memo) for f in f_sym]
            print("Loaded taylor dynamics from file!")
        else:
            f_val = [TaylorExpand(f, {z[i]: z0[i] for i in range(len(z))}, order) for f in f_val]
            print("Finished taylor expansion!")

            memo = dict()
            f_sym = [pydrake.all.to_sympy(f, memo=memo) for f in f_val]
            np.save(filepath, f_sym)
            print("Saved taylor dynamics to file!")

        return f_val
    def f2(z, dtype=Expression):
        sph = np.sin(z[1])
        cph = np.cos(z[1])
        st = np.sin(z[2])
        ct = np.cos(z[2])
        sps = np.sin(z[3])
        cps = np.cos(z[3])
        x_dot = z[4]
        phi_dot = z[5]
        theta_dot = z[6]
        psi_dot = z[7]

        assert len(z) == nz
        denom1 = ct ** 2
        denom2 = m1 + m2 * st ** 2

        f2_val = np.zeros([nx, nu], dtype=dtype)
        f2_val[4, :] = [1/denom2, sps*st*ct/(L*denom2)]  # xddot
        f2_val[6, :] = [-ct/(L*denom2), -(m1+m2)*sps*st/(m2*L**2*denom2)]  # thetaddot
        f2_val[7, :] = [0, cps/(m2*L**2*denom1)]  # psiddot
        return f2_val


    # Define State Limits
    # x = (x, theta, psi, xdot, thetadot, psidot)
    # z = (x, st, ct, sp, cp, xdot, thetadot, psidot)
    u_max = np.array([500, 2000])
    z_max = np.array([1, np.pi/4, np.pi/2, np.pi/2, 1, 1, 1, 1])
    z_min = np.array([-1, -np.pi/4, -np.pi/2, -np.pi/2, -1, -1, -1, -1])
    assert (z_min < z_max).all()

    # Equilibrium point in both the system coordinates.
    x0 = np.zeros(nx)
    z0 = x2z(x0)
    z0[np.abs(z0) <= 1e-6] = 0

    # Quadratic running cost in augmented state.
    # state weighting matrix
    # z = (x, phi, theta, psi, xdot, phidot, thetadot, psidot)
    Q_diag = [0.0, 0, 20000, 20000000, 0.0, 0, 1, 1]

    Q = np.diag(Q_diag)
    # u = (fx fy)
    # control weighting matrix
    R = np.array([[.1, 0.0], [0.0, 0.1]])

    def l_cost(z, u):
        zbar = z-z0
        # zbar[0] = 0
        # zbar [4] = 0
        return zbar.dot(Q).dot(zbar) + u.dot(R).dot(u)
    Rinv = np.linalg.inv(R)

    if test:
        return nz, f, f2, x2z, Rinv, z0


    # Set up optimization.
    ctrl_vars = [0,1,2,3,4,5,6,7]
    prog = MathematicalProgram()
    z = prog.NewIndeterminates(nz, "z")
    u = prog.NewIndeterminates(nu, "u")
    zu = np.concatenate((z, u))
    J = prog.NewFreePolynomial(Variables(z[ctrl_vars]), deg)
    J_expr = J.ToExpression()

    if read_file:
        filename = read_file
        J_star = load_polynomial(z, filename+".pkl")
        Rinv = np.linalg.inv(R)
        f2_val = f2(z)
        dJdz = J_star.Jacobian(z)
        u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)
        uToStr(u_star, filename+".txt")
        plot_value_function(J_star, z, z_max, u_max, 0, 0, plot_states="thetapsi", actuator_saturate=False)
        return 0


    # Configure Optimization Strategy
    if constraint == "kSos":
        constraint_type = prog.NonnegativePolynomial.kSos
    elif constraint == "kSdsos":
        constraint_type = prog.NonnegativePolynomial.kSdsos
    else:
        constraint_type = prog.NonnegativePolynomial.kDsos


    obj = J
    for i in ctrl_vars: #range(nz):
        obj = obj.Integrate(z[i], z_min[i], z_max[i])
    prog.AddCost(-obj.ToExpression())

    # Enforce Lyapunov function is negative definite (HJB inequality)
    f_val = f(z, u)
    J_dot = J_expr.Jacobian(z).dot(f_val)
    LHS = J_dot + l_cost(z, u)   # Lower bound on cost to go V >= -l  -->  V + l >= 0
    # LHS = J_dot + 3 # Relaxed Hamilton jacobian bellman conditions, non-optimal, but still lyapunov

    ring_deg = deg# Polynomial(LHS).TotalDegree() - 2
    # z = (x, st, ct, sp, cp, xdot, thetadot, psidot)

    # S procedure for z_min < z < z_max
    S_Jdot = 0
    for i in ctrl_vars: #np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(zu), int(np.ceil(ring_deg/2)*2), type=constraint_type)[0].ToExpression()
        S_Jdot += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])  # negative inside the range of z-space we are defining to be locally stable (X^h set in paper)

    # Enforce Input constraint
    u_min = -u_max
    if actuator_saturate:
        for i in range(nu):
            lam = prog.NewSosPolynomial(Variables(zu), int(np.ceil(ring_deg/2)*2), type=constraint_type)[0].ToExpression()
            S_Jdot += lam * (u[i] - u_max[i]) * (u[i] - u_min[i])
    prog.AddSosConstraint(LHS + S_Jdot, type=constraint_type)
    # prog.AddSosConstraint(LHS, type=constraint_type)


    # Enforce that value function nonnegativity constraint for all x in set X^h (positive semidefinite req)
    S_J = 0
    # S procedure for z_min < z < z_max
    for i in ctrl_vars: #np.arange(nz):
        lam = prog.NewSosPolynomial(Variables(z), int(np.ceil(deg/2)*2), type=constraint_type)[0].ToExpression()
        S_J += lam * (z[i] - z_max[i]) * (z[i] - z_min[i])  # +ve when z > zmax or z < zmin, -ve inside z bounds
    prog.AddSosConstraint(J_expr + S_J, type=constraint_type)
    # prog.AddSosConstraint(J_expr, type=constraint_type)


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
    f2_val = f2(z)
    dJdz = J_star.ToExpression().Jacobian(z)
    u_star = -0.5 * Rinv.dot(f2_val.T).dot(dJdz.T)

    # Save data to file
    os.makedirs("Sk8Dyn/data/Taylor/{}/{}".format(z_max, constraint), exist_ok=True)
    save_polynomial(J_star, z, "Sk8Dyn/data/Taylor/{}/{}/J_lower_bound_deg_{}.pkl".format(z_max, constraint, deg))
    uToStr(u_star, "Sk8Dyn/data/Taylor/{}/{}/J_lower_bound_deg_{}.txt".format(z_max, constraint, deg))

    if visualize:
        Vx =10
        plot_value_function(J_star, z, z_max, u_max, 0, Vx, plot_states="thetapsi", actuator_saturate=actuator_saturate)

    return J_star, z

def plot_value_function(J_star, z, z_max, u_max, phi, Vx, plot_states="xy", actuator_saturate=False):
    nz = 8
    x_max = z_max
    x_max[1] -= 0.1
    x_max[2] -= 0.1
    x_min = -x_max

    try:
        dJdz = J_star.ToExpression().Jacobian(z)
    except:
        dJdz = J_star.Jacobian(z)

    zero_vector = np.zeros(51*51)
    phi_vector = np.ones(51*51)*phi
    xdot_vector = np.ones(51*51)*Vx
    # x = (x, phi, theta, psi, xdot, phidot, thetadot, psidot)
    if plot_states == "xtheta":
        X1, THETA = np.meshgrid(np.linspace(x_min[0], x_max[0], 51),
                        np.linspace(x_min[2], x_max[2], 51))
        X = np.vstack((X1.flatten(), phi_vector, THETA.flatten(), zero_vector, xdot_vector, zero_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, X, X1, THETA, x_min, x_max, z_max, u_max, 0, 2, "x", "theta", actuator_saturate=actuator_saturate)
    elif plot_states == "thetapsi":
        THETA1, PSI = np.meshgrid(np.linspace(x_min[2], x_max[2], 51),
                        np.linspace(x_min[3], x_max[3], 51))
        THETA = np.vstack((zero_vector, phi_vector, THETA1.flatten(), PSI.flatten(), xdot_vector, zero_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, THETA, THETA1, PSI, x_min, x_max, z_max, u_max, 2, 3, "theta", "psi", actuator_saturate=actuator_saturate)
    elif plot_states == "thetadot":
        THETA1, THETADOT = np.meshgrid(np.linspace(x_min[2], x_max[2], 51),
                        np.linspace(x_min[6], x_max[6], 51))
        THETA = np.vstack((zero_vector, phi_vector, THETA1.flatten(), zero_vector, zero_vector, xdot_vector, THETADOT.flatten(), zero_vector))
        plotCostPolicy(dJdz, J_star, z, THETA, THETA1, THETADOT, x_min, x_max, z_max, u_max, 2, 6, "theta", "theta_dot", actuator_saturate=actuator_saturate)
    elif plot_states == "psidot":
        PSI1, PSIDOT = np.meshgrid(np.linspace(x_min[3], x_max[3], 51),
                        np.linspace(x_min[7], x_max[7], 51))
        PSI = np.vstack((zero_vector, phi_vector, zero_vector, PSI1.flatten(), xdot_vector, zero_vector, zero_vector, PSIDOT.flatten()))
        plotCostPolicy(dJdz, J_star, z, PSI, PSI1, PSIDOT, x_min, x_max, z_max, u_max, 3, 7, "psi", "psi_dot", actuator_saturate=actuator_saturate)
    elif plot_states == "phipsi":
        PHI1, PSI = np.meshgrid(np.linspace(x_min[1], x_max[1], 51),
                            np.linspace(x_min[3], x_max[3], 51))
        PHI = np.vstack((zero_vector, PHI1.flatten(), zero_vector, PSI.flatten(), xdot_vector, zero_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, PHI, PHI1, PSI, x_min, x_max, z_max, u_max, 1, 3, "phi", "psi", actuator_saturate=actuator_saturate)

    elif plot_states == "phitheta":
        PHI1, THETA = np.meshgrid(np.linspace(x_min[1], x_max[1], 51),
                            np.linspace(x_min[2], x_max[2], 51))
        PHI = np.vstack((zero_vector, PHI1.flatten(), THETA.flatten(), zero_vector, xdot_vector, zero_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, PHI, PHI1, THETA, x_min, x_max, z_max, u_max, 1, 2, "phi", "theta", actuator_saturate=actuator_saturate)

    elif plot_states == "phixdot":
        PHI1, XDOT = np.meshgrid(np.linspace(x_min[1], x_max[1], 51),
                            np.linspace(x_min[4], x_max[4], 51))
        PHI = np.vstack((zero_vector, PHI1.flatten(), zero_vector, zero_vector, XDOT.flatten(), zero_vector, zero_vector, zero_vector))
        plotCostPolicy(dJdz, J_star, z, PHI, PHI1, XDOT, x_min, x_max, z_max, u_max, 1, 4, "phi", "x_dot", actuator_saturate=actuator_saturate)


def plotCostPolicy(dJdz, J_star, z, X, X1, X2, x_min, x_max, z_max, u_max, xaxis_ind, yaxis_ind, xlabel, ylabel, actuator_saturate=False):
    nz, f, f2, x2z, Rinv, z0 = sk8dyn_sos_lower_bound(2, test=True)
    Z = x2z(X)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # figsize can be adjusted as needed
    for i in range(2):

        J = np.zeros(Z.shape[1])
        U = np.zeros(Z.shape[1])
        for _ in range(Z.shape[1]):
            z_val = Z[:, _]
            J[_] = J_star.Evaluate(dict(zip(z, z_val)))
            f2_val = f2(z_val, dtype=float)
            dJdz_val = np.zeros(nz)
            for n in range(nz):
                dJdz_val[n] = dJdz[n].Evaluate(dict(zip(z, z_val)))
            U[_] = calc_u_opt(dJdz_val, f2_val, Rinv)[i]
            if actuator_saturate:
                U[_] = np.clip(U[_], -u_max[i], u_max[i])
            else:
                continue
                # U[_] = np.clip(U[_], -2000, 2000)

        # axs[i,0].remove()
        # axs[i, 0] = fig.add_subplot(2,2,2*i+1, projection='3d')
        axs[i,0].set_xlabel(xlabel)
        axs[i,0].set_ylabel(ylabel)
        axs[i,0].set_title(f"Cost-to-Go, Input #: {i}")
        im = axs[i,0].imshow(J.reshape(X1.shape),
                       cmap=cm.jet, aspect='auto',
                       extent=(x_min[xaxis_ind], x_max[xaxis_ind], x_max[yaxis_ind], x_min[yaxis_ind]))
        axs[i,0].invert_yaxis()
        fig.colorbar(im, ax=axs[i,0])
        # im = axs[i,0].plot_surface(X1, X2, J.reshape(X1.shape), cmap=cm.coolwarm)  # Use 'contour' for line contours
        # fig.colorbar(im, ax=axs[i, 0])

        axs[i,1].set_xlabel(xlabel)
        axs[i,1].set_ylabel(ylabel)
        axs[i,1].set_title(f"Policy, Input #: {i}")
        if i==0:
            im = axs[i,1].imshow(U.reshape(X1.shape),
                           cmap=cm.jet, aspect='auto',
                           extent=(x_min[xaxis_ind], x_max[xaxis_ind], x_max[yaxis_ind], x_min[yaxis_ind]))
            axs[i,1].invert_yaxis()
            fig.colorbar(im, ax=axs[i,1])
        elif i == 1:
            # axs[i,1].contour(X1, X2, U.reshape(X1.shape), levels = np.linspace(-2000, 2000, 50), extent=(x_min[xaxis_ind], x_max[xaxis_ind], x_max[yaxis_ind], x_min[yaxis_ind]))
            im = axs[i,1].contourf(X1, X2, U.reshape(X1.shape), levels=np.linspace(-3000, 3000, 51), cmap=cm.jet, extend="both")
            # axs[i, 1].invert_yaxis()
            fig.colorbar(im, ax=axs[i, 1])

    plt.tight_layout()
    plt.show()

def uToStr(U, file=None):
    # Define symbolic variables
    fx = U[0]
    fy = U[1]
    # Map z matrix parameters to symbolic representations
    fx_str_sub = fx.__str__().replace("z(0)", "x") \
        .replace("z(1)", "phi") \
        .replace("z(2)", "theta") \
        .replace("z(3)", "psi") \
        .replace("z(4)", "x_dot") \
        .replace("z(5)", "phi_dot") \
        .replace("z(6)", "theta_dot") \
        .replace("z(7)", "psi_dot")
    fy_str_sub = fy.__str__().replace("z(0)", "x") \
        .replace("z(1)", "phi") \
        .replace("z(2)", "theta") \
        .replace("z(3)", "psi") \
        .replace("z(4)", "x_dot") \
        .replace("z(5)", "phi_dot") \
        .replace("z(6)", "theta_dot") \
        .replace("z(7)", "psi_dot")
    # fx_str_sub= re.sub(r"pow\(([^,]+),\s*(\d+)\)", r"\1^\2", fx_str_sub)
    # fy_str_sub = re.sub(r"pow\(([^,]+),\s*(\d+)\)", r"\1^\2", fy_str_sub)
    print(f"fx: {fx_str_sub}")
    print(f"fy: {fy_str_sub}")

    # Save to text file for Transfer to Matlab
    if file:
        with open(file, "w") as text_file:
            text_file.write(fx_str_sub+"\n"+fy_str_sub)


# filename = "/home/thomas/Documents/thesis/LyapunovConvexOptimization/SphericalIP/data/[1.5 1.5 1.  1.  1.  1.  4.  4.  3.  3. ]/J_lower_bound_deg_4_SOS_Q20000"
filename = "/home/thomas/Documents/thesis/LyapunovConvexOptimization/Sk8Dyn/data/Taylor/J_lower_bound_deg_6"
J_star, z = sk8dyn_sos_lower_bound(4, constraint = "kSos", visualize=True, actuator_saturate=False, read_file=None)
