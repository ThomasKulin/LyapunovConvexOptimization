import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mpld3
from IPython.display import Markdown, display
from pydrake.all import (
    MathematicalProgram,
    Solve,
    SymbolicVectorSystem,
    ToLatex,
    Variables,
)
from pydrake.symbolic import Polynomial

from underactuated import plot_2d_phase_portrait, running_as_notebook

def sos_verify():
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(2, "x")
    f = [-x[0] - 2 * x[1] ** 2,
         -x[1] - x[0] * x[1] - 2 * x[1] ** 3]

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
    # display(
    #     Markdown(
    #         "$V(x) = "
    #         + ToLatex(
    #             Polynomial(result.GetSolution(V))
    #             .RemoveTermsWithSmallCoefficients(1e-5)
    #             .ToExpression(),
    #             6,
    #         )
    #         + "$"
    #     )
    # )


# sos_optimize()


def sos_optimize_morevars():
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(2, "x")
    f = [-x[0] - 2 * x[1] ** 2, -x[1] - x[0] * x[1] - 2 * x[1] ** 4]

    sys = SymbolicVectorSystem(state=x, dynamics=f)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_2d_phase_portrait(sys, (-3, 3), (-3, 3))
    plt.show()

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



sos_optimize_morevars()
