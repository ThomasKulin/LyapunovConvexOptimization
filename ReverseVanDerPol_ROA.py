# python libraries
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

# pydrake imports
from pydrake.all import (LinearQuadraticRegulator, MathematicalProgram,
                         RealContinuousLyapunovEquation, MonomialBasis, Solve, Polynomial, Variables, Monomial)
from pydrake.examples import VanDerPolOscillator
# underactuated imports
from underactuated import plot_2d_phase_portrait

# increase default size matplotlib figures
rcParams['figure.figsize'] = (6, 6)

# function that implements the time-reversed Van der Pol dynamics
f = lambda x: [- x[1], x[0] + (x[0]**2 - 1) * x[1]]

# Define P same way as in the class DeepNote.
A = np.array([[0, -1], [1, -1]])
Q = np.eye(2)
P = RealContinuousLyapunovEquation(A, Q)
prog = MathematicalProgram()

x = prog.NewIndeterminates(2, "x")

V = x.dot(P).dot(x)
V_dot = 2 * x.dot(P).dot(f(x))

l_deg = 4
l_poly, l_gram = prog.NewSosPolynomial(Variables(x), l_deg)
l = l_poly.ToExpression()

rho = prog.NewContinuousVariables(1)

x_normsq = x.dot(x)
prog.AddSosConstraint(x_normsq * (V - rho[0]) - l * V_dot)

prog.AddLinearCost(-rho[0])

result = Solve(prog)
rho_sol = result.GetSolution(rho)[0]
l_sol_gram = result.GetSolution(l_gram)

x_old = x

print(result.is_success())
print(f'rho = {rho_sol}.')

prog1 = MathematicalProgram()
x = prog1.NewIndeterminates(2, "x")

V = x.dot(P).dot(x)
V_dot = 2 * x.dot(P).dot(f(x))

l_deg = 0
l_poly = prog1.NewFreePolynomial(Variables(x), l_deg)
l = l_poly.ToExpression()

rho1 = prog1.NewContinuousVariables(1)

x_normsq = x.dot(x)
prog1.AddSosConstraint(x_normsq * (V - rho1[0]) - l * V_dot)

prog1.AddLinearCost(-rho1[0])

result = Solve(prog1)
rho_sol_mod = result.GetSolution(rho1)[0]

x_old_mod = x

print(result.is_success())
print(f'rho = {rho_sol_mod}.')

prog = MathematicalProgram()

x = prog.NewIndeterminates(2, "x")

V = x.dot(P).dot(x)
V_dot = 2 * x.dot(P).dot(f(x))

l_poly, l_gram = prog.NewSosPolynomial(Variables(x), l_deg)
l = l_poly.ToExpression()

rho = prog.NewContinuousVariables(1)

VDOT_FEAS_EPS = 3e-1

x_normsq = x.dot(x)
# prog.AddSosConstraint(x_normsq * (V - rho[0]) - l * (V_dot + VDOT_FEAS_EPS * x_normsq))
prog.AddSosConstraint(x_normsq * (V - rho[0]) - l * (V_dot + VDOT_FEAS_EPS * x_normsq))


prog.AddLinearCost(-rho[0])

result = Solve(prog)
rho_sol_harder = result.GetSolution(rho)[0]
l_sol_harder = result.GetSolution(l)

x_old_harder = x

print(result.is_success())
print(f'rho = {rho_sol} > rho_harder = {rho_sol_harder}.')
assert rho_sol_harder < rho_sol

print("l_sol_harder: ", l_sol_harder)
l_sol_harder_pruned = Polynomial(l_sol_harder).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()
print("l_sol_harder_pruned: ", l_sol_harder_pruned)


def plot_V(rho, label, color, linestyle):
    # grid of the state space
    x1 = np.linspace(*xlim)
    x2 = np.linspace(*xlim)
    X1, X2 = np.meshgrid(x1, x2)

    # function that evaluates V(x) at a given x
    # (looks bad, but it must accept meshgrids)
    eval_V = lambda x: sum(sum(x[i] * x[j] * Pij for j, Pij in enumerate(Pi)) for i, Pi in enumerate(P))

    # contour plot with only the rho level set
    cs = plt.contour(X1, X2, eval_V([X1, X2]), levels=[rho], colors=color, linestyle=linestyle, linewidths=3, zorder=3)

    # misc plot settings
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.gca().set_aspect('equal')

    # fake plot for legend
    plt.plot(0, 0, color=color, linewidth=3, label=label)
    plt.legend()
    return cs


# function that plots the levels sets of Vdot(x)
def plot_Vdot():
    # grid of the state space
    x1 = np.linspace(*xlim)
    x2 = np.linspace(*xlim)
    X1, X2 = np.meshgrid(x1, x2)

    # function that evaluates Vdot(x) at a given x
    eval_Vdot = lambda x: 2 * sum(sum(x[i] * f(x)[j] * Pij for j, Pij in enumerate(Pi)) for i, Pi in enumerate(P))

    # contour plot with only the rho level set
    cs = plt.contour(X1, X2, eval_Vdot([X1, X2]), colors='b', levels=np.linspace(-10, 40, 11))
    plt.gca().clabel(cs, inline=1, fontsize=10)

    # misc plot settings
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.gca().set_aspect('equal')

    # fake plot for legend
    plt.plot(0, 0, color='b', label=r'$\dot{V}(\mathbf{x})$')
    plt.legend()
    return cs


xlim = (-3, 3)
limit_cycle = VanDerPolOscillator.CalcLimitCycle()

plot_2d_phase_portrait(f, x1lim=xlim, x2lim=xlim)
plot_V(rho_sol, "original", "r", linestyle='solid')
plot_V(rho_sol_harder, "harder", "g", linestyle='solid')
plot_V(rho_sol_mod, "SOS relaxed", "m", linestyle='dashed')
plt.plot(limit_cycle[0], limit_cycle[1], color='b', linewidth=3, label='ROA boundary')
plt.legend(loc=1)
plt.show()

fig, ax = plt.subplots()
plot_Vdot()
plot_V(rho_sol, "original", "r", linestyle='solid')
plot_V(rho_sol_harder, "harder", "g", linestyle='solid')
plot_V(rho_sol_mod, "SOS relaxed", "m", linestyle='dashed')
plt.show()