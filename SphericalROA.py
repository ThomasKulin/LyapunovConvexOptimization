import numpy as np
from pydrake.all import (
    Jacobian,
    MathematicalProgram,
    RegionOfAttraction,
    Solve,
    SymbolicVectorSystem,
    Variable,
    Variables,
)

def sos_verify_known_roa():
    prog = MathematicalProgram()
    # Variables are [sin(theta) cos(theta) theta_dot sin(psi) cos(psi) psi_dot]

    st = prog.NewIndeterminates(1, "st")[0]
    ct = prog.NewIndeterminates(1, "ct")[0]
    sp = prog.NewIndeterminates(1, "sp")[0]
    cp = prog.NewIndeterminates(1, "cp")[0]
    thetadot = prog.NewIndeterminates(1, "\dot\\theta")[0]
    psidot = prog.NewIndeterminates(1, "\dot\\psi")[0]
    x = np.array([st, ct, thetadot, sp, cp, psidot])

    g = 9.81
    Len = 0.9
    m2 = 90

    # Map from original state to augmented state.
    # Uses sympy to be able to do symbolic integration later on.
    # x = (theta, thetadot, phi, phidot)
    # z = (st, ct, thetadot, sp, cp, psidot)
    x2z = lambda x: np.array([np.sin(x[0]), np.cos(x[0]), x[1], np.sin(x[2]), np.cos(x[2]), x[3]])

    # Write out the dynamics in terms of sin(theta/psi), cos(theta/psi), and theta/psi dot
    f = [
        ct * thetadot,
        -st * thetadot,
        -st * (g * cp + Len * ct * psidot ** 2) / Len,
        cp * psidot,
        -sp * psidot,
        (-g * sp + 2 * Len * psidot * thetadot * st) / (Len * ct)
    ]

    # The fixed-point in this coordinate (because cos(0)=1).
    x0 = np.array([0, 1, 0, 0, 1, 0])

    # Define the dynamics and Lyapunov function.
    V = x.dot(x)
    Vdot = Jacobian([V], x).dot(f)[0]
    rho = 1

    # Define the Lagrange multiplier.
    deg_L = 2
    L = prog.NewSosPolynomial(Variables(x), deg_L)[0].ToExpression()


    # prog.AddSosConstraint(-Vdot + lambda_ * (V - rho))
    prog.AddSosConstraint(
        -Vdot - L * (V - rho)
    )

    result = Solve(prog)

    assert result.is_success(), "Optimization failed"

    print("Verified that " + str(V) + " < 1 is in the region of attraction.")


sos_verify_known_roa()

def sos_maximize_roa():
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(1, "x")
    rho = prog.NewContinuousVariables(1, "rho")[0]

    # Define the dynamics and Lyapunov function.
    f = -x + x**3
    V = x.dot(x)
    Vdot = Jacobian([V], x).dot(f)[0]

    # Define the Lagrange multiplier.
    lambda_ = prog.NewFreePolynomial(Variables(x), 0).ToExpression()

    prog.AddSosConstraint((V - rho) * x.dot(x) - lambda_ * Vdot)
    prog.AddLinearCost(-rho)

    result = Solve(prog)

    assert result.is_success()

    print(
        "Verified that "
        + str(V)
        + " < "
        + str(result.GetSolution(rho))
        + " is in the region of attraction."
    )

    assert np.fabs(result.GetSolution(rho) - 1) < 1e-5

# sos_maximize_roa()
