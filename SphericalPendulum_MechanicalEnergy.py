import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mpld3
import drake

import numpy as np
from IPython.display import Markdown, display
from pydrake.all import MathematicalProgram, Solve, ToLatex, Variables, MosekSolver, SolverOptions, CommonSolverOption
from pydrake.examples import PendulumParams
from pydrake.symbolic import Polynomial
import os

from underactuated import running_as_notebook

from underactuated import plot_2d_phase_portrait, running_as_notebook

if "MOSEKLM_LICENSE_FILE" not in os.environ:
    # If a mosek.lic file has already been uploaded, then simply use it here.
    if os.path.exists('/home/thomas/mosek/mosek.lic'):
        os.environ["MOSEKLM_LICENSE_FILE"] = "/home/thomas/mosek/mosek.lic"
    else:
        print("stop bein a loooooser and get mosek")

print(MosekSolver().enabled())

def global_pendulum():
    prog = MathematicalProgram()

    # Declare the "indeterminates", x.  These are the variables which define the
    # polynomials, but are NOT decision variables in the optimization.  We will
    # add constraints below that must hold FOR ALL x.

    # Variables are [sin(theta) cos(theta) theta_dot sin(psi) cos(psi) psi_dot]

    st = prog.NewIndeterminates(1, "st")[0]
    ct = prog.NewIndeterminates(1, "ct")[0]
    sp = prog.NewIndeterminates(1, "sp")[0]
    cp = prog.NewIndeterminates(1, "cp")[0]
    thetadot = prog.NewIndeterminates(1, "\dot\\theta")[0]
    psidot = prog.NewIndeterminates(1, "\dot\\psi")[0]
    x = np.array([st, ct, thetadot, sp, cp, psidot])

    g =9.81
    Len = 0.5
    m2 = 1


    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (theta, thetadot, phi, phidot)
    # z = (st, ct, thetadot, sp, cp, psidot)
    x2z = lambda x: np.array([np.sin(x[0]), np.cos(x[0]), x[1], np.sin(x[2]), np.cos(x[2]), x[3]])

    # Write out the dynamics in terms of sin(theta/psi), cos(theta/psi), and theta/psi dot
    denom = (Len*ct)
    f = [
        ct * thetadot * denom,
        -st * thetadot * denom,
        -st*(g*cp + Len*ct*psidot**2)/Len * denom,
        cp * psidot * denom,
        -sp * psidot * denom,
        (-g*sp + 2*Len*psidot*thetadot*st)
    ]


    # The fixed-point in this coordinate (because cos(0)=1).
    x0 = np.array([0, 1, 0, 0, 1, 0])

    # Construct a polynomial V that contains all monomials with s,c,thetadot up
    # to degree 2.
    # TODO(russt): use the Groenber basis for s²+c²=1.
    deg_V = 4
    V = prog.NewFreePolynomial(Variables(x), deg_V).ToExpression()

    # Add a constraint to enforce that V is strictly positive away from x0.
    # (Note that because our coordinate system is sine and cosine, V is also
    # zero at theta=2pi, etc).
    eps = 1e-4
    prog.AddSosConstraint(V - eps * (x - x0).dot(x - x0))

    # Construct the polynomial which is the time derivative of V.
    Vdot = V.Jacobian(x).dot(f)
    temp = Polynomial(Vdot).TotalDegree()

    # Construct a polynomial L representing the "Lagrange multiplier".
    deg_L = 4
    L = prog.NewFreePolynomial(Variables(x), deg_L).ToExpression()

    # Add a constraint that Vdot is strictly negative away from x0 (but make an
    # exception for x-axis fixed point by multiplying by ct^2).
    constraint2 = prog.AddSosConstraint(
        # -Vdot - (L * (st**2 + ct**2 + sp**2 + cp**2 - 2) - eps * (x - x0).dot(x - x0) * (ct)**2))
    -Vdot - (L * (st ** 2 + ct ** 2 * sp ** 2 + cp ** 2 * ct ** 2- 1) - eps * (x - x0).dot(x - x0) * (ct) ** 2))

    # TODO(russt): When V is the mechanical energy, Vdot=-b*thetadot^2, so I may not need all of the multipliers here.

    # Add V(0) = 0 constraint
    constraint3 = prog.AddLinearConstraint(V.Substitute({st: 0, ct: 1, thetadot: 0, sp: 0, cp: 1, psidot: 0}) == 0)

    # Add V(theta=pi) = mgl, just to set the scale.
    constraint4 = prog.AddLinearConstraint(
        V.Substitute({st: 1, ct: 0, thetadot: 0, sp: 1, cp: 0, psidot: 0}) == m2*g*Len
    )

    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)

    # Call the solver.
    result = Solve(prog)
    assert result.is_success()

    # Note that I've added mgl to the potential energy (relative to the textbook),
    # so that it would be non-negative... like the Lyapunov function.
    mgl = m2*g*Len
    print(f" NOT GOOD $E = {0.5*m2 * Len**2} \dot\\theta^2 + {mgl}(1-c)$\n")


    Vsol = Polynomial(result.GetSolution(V))
    print(f'V = {Vsol.RemoveTermsWithSmallCoefficients(1e-3).ToExpression()}')


    # Plot the results as contour plots.
    nq = 151
    nqd = 151
    q = np.linspace(-2 * np.pi, 2 * np.pi, nq)
    qd = np.linspace(-2 * mgl, 2 * mgl, nqd)
    Q, QD = np.meshgrid(q, qd)
    Energy = 0.5 * m2* Len ** 2 * QD**2 + mgl * (1 - np.cos(Q))
    Vplot = Q.copy()
    env = {st: np.sin(np.pi/2), ct: np.sin(np.pi/2), thetadot: 0, sp: np.sqrt(2)/2, cp: np.sqrt(2)/2, psidot: 0}
    for i in range(nq):
        for j in range(nqd):
            env[st] = np.sin(Q[i, j])
            env[ct] = np.cos(Q[i, j])
            env[thetadot] = QD[i, j]
            Vplot[i, j] = Vsol.Evaluate(env)

    # plt.rc("text", usetex=True)
    fig, ax = plt.subplots()
    ax.contour(Q, QD, Vplot)
    # ax.contour(Q, QD, Energy, alpha=0.5, linestyles="dashed")
    ax.set_xlabel("theta")
    ax.set_ylabel("thetadot")
    ax.set_title("V (solid) and Mechanical Energy (dashed)")

    plt.show()


global_pendulum()