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
    x = prog.NewIndeterminates(1, "x")

    # Define the dynamics and Lyapunov function.
    f = -x + x**3
    V = x.dot(x)
    Vdot = Jacobian([V], x).dot(f)[0]
    rho = 1

    # Define the Lagrange multiplier.
    lambda_ = prog.NewSosPolynomial(Variables(x), 2)[0].ToExpression()

    prog.AddSosConstraint(-Vdot + lambda_ * (V - rho))

    result = Solve(prog)

    assert result.is_success(), "Optimization failed"

    print("Verified that " + str(V) + " < 1 is in the region of attraction.")


# sos_verify_known_roa()

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

sos_maximize_roa()
