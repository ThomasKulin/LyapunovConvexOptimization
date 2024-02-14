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

def sos_verify():
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

    g = 9.81
    Len = 0.5
    m2 = 1

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (theta, thetadot, phi, phidot)
    # z = (st, ct, thetadot, sp, cp, psidot)
    x2z = lambda x: np.array([np.sin(x[0]), np.cos(x[0]), x[1], np.sin(x[2]), np.cos(x[2]), x[3]])

    # Write out the dynamics in terms of sin(theta/psi), cos(theta/psi), and theta/psi dot
    denom = (Len * ct)
    f = [
        ct * thetadot * denom,
        -st * thetadot * denom,
        st * (g * cp - Len * ct * psidot ** 2) / Len * denom,
        cp * psidot * denom,
        -sp * psidot * denom,
        (g * sp + 2 * Len * psidot * thetadot * st)
    ]

    # The fixed-point in this coordinate (because cos(0)=1).
    x0 = np.array([0, 1, 0, 0, 1, 0])


    V = x[0] ** 2 + 2 * x[1] ** 2
    Vdot = V.Jacobian(x).dot(f)

    prog.AddSosConstraint(-Vdot)

    result = Solve(prog)
    assert result.is_success()

    sys = SymbolicVectorSystem(state=x, dynamics=f)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_2d_phase_portrait(sys, (-3, 3), (-3, 3))
    print("Successfully verified Lyapunov candidate")
    plt.show()

# sos_verify()

def sos_optimize():
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(2, "x")
    f = [-x[0] - 2 * x[1] ** 2, -x[1] - x[0] * x[1] - 2 * x[1] ** 3]

    V = prog.NewSosPolynomial(Variables(x), 2)[0].ToExpression()
    print("Candidate:")
    display(Markdown("$V(x) = " + ToLatex(V) + "$"))
    prog.AddLinearConstraint(V.Substitute({x[0]: 0, x[1]: 0}) == 0)
    prog.AddLinearConstraint(V.Substitute({x[0]: 1, x[1]: 0}) == 1)
    Vdot = V.Jacobian(x).dot(f)

    prog.AddSosConstraint(-Vdot)

    result = Solve(prog)
    assert result.is_success()

    print("Solution:")
    print(f'V(x) = {Polynomial(result.GetSolution(V)).RemoveTermsWithSmallCoefficients(1e-5).ToExpression()}')


# sos_optimize()

