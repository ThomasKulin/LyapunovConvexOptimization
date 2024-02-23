# import sys
# sys.path.append(
#     "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
from pydrake.all import (MathematicalProgram, Polynomial, Expression,
                         SolverOptions, CommonSolverOption,
                         SolutionResult, Variables, Solve, Jacobian, Evaluate,
                         Substitute, MosekSolver, MonomialBasis)
import math
import numpy as np
from numpy.linalg import eig, inv
import cvxpy as cp
from math import factorial as fact
from veril.util.plots import *
import time


def global_vdot(system, V, Vdot, circle=None):
    assert system.loop_closed
    x = system.sym_x
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)

    degV = Polynomial(V, x).TotalDegree()
    degL1 = degV - 1 + system.degf
    degVdot = Polynomial(Vdot, x).TotalDegree()
    recast_cstr = 0
    if hasattr(system, 'recast'):
        L2 = prog.NewFreePolynomial(Variables(x), degL1).ToExpression()
        recast_cstr = L2 * system.recast
    rho = prog.NewContinuousVariables(1, "r")[0]
    # prog.AddConstraint(rho >= 0)
    L1 = prog.NewFreePolynomial(Variables(x), degL1).ToExpression()
    prog.AddSosConstraint(-Vdot + recast_cstr - rho)

    prog.AddCost(-rho)
    solver = MosekSolver()

    # solver.set_stream_logging(True, '')
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = solver.Solve(prog, None, None)
    print(result.get_solution_result())
    # print('w/ solver %s' % (result.get_solver_id().name()))
    assert result.is_success()
    L1 = result.GetSolution(L1)
    rho = result.GetSolution(rho)
    print(result.GetSolution(rho))
    # P = result.GetSolution(P)
    # print('cond # of gram is %s' % np.linalg.cond(P))


def verify_via_equality(system, V0):
    assert system.loop_closed
    if V0 is None:
        A, S, V0 = system.linearized_quadractic_V()
    start = time.time()
    V = levelset_sos(system, V0, do_balance=False)
    # V= levelset_w_feature_transformation(system, None, )
    end = time.time()
    print('equlity constrained time %s' % (end - start))
    plot_funnel(V, system, slice_idx=system.slice, add_title=' - Equality ' +
                'Constrainted Result')
    return V


def verify_via_bilinear(system, **kwargs):
    assert system.loop_closed
    sys_name = system.name
    degV = system.degV
    degf = system.degf
    degL = degV - 1 + degf
    options = {'degV': degV, 'do_balance': False, 'degL1': degL, 'degL2': degL,
               'converged_tol': 2e-2, 'max_iterations': 20}

    start = time.time()
    A, S, V0 = system.linearized_quadractic_V()

    if 'V0' in kwargs:
        V0 = kwargs['V0']
    V = bilinear(
        system.sym_x, V0, system.sym_f, S, A, **options)
    end = time.time()
    print('bilinear time %s' % (end - start))
    plot_funnel(V, system, slice_idx=system.slice,
                add_title=' - Bilinear Result')
    return system, V


def cvx_V(system, sys_name):
    tag = str(system.deg_ftrs)
    model_dir = '../data/' + sys_name
    train_x = np.load(model_dir + '/stableSamplesSlice12.npy')
    num_samples = train_x.shape[0]
    assert(train_x.shape[1] == system.num_states)
    print('x size %s' % str(train_x.shape))

    file_path = model_dir + '/features_' + tag + '.npz'

    if os.path.exists(file_path):
        l = np.load(file_path)
        features = [l['phi'], l['eta']]
    else:
        features = system.features_at_x(train_x, file_path)
        np.savez_compressed(file_path, phi=features[0], eta=features[1])
    assert(features[0].shape[0] == num_samples)
    P = convexly_search_for_V_on_samples(features)
    cvx_P_filename = model_dir + '/cvx_P_' + str(system.deg_ftrs) + '.npy'
    np.save(cvx_P_filename, P)
    V, Vdot = system.P_to_V(P)
    return V, Vdot, system


def bilinear(x, V0, f, S0, A, **kwargs):
    V = V0
    nX = x.shape[0]
    do_balance = kwargs['do_balance']
    max_iterations = kwargs['max_iterations']
    converged_tol = kwargs['converged_tol']
    if do_balance:
        [T, V0bal, fbal, S0, A] = balance(x, V0, f, S0, A)
    else:
        T, V0bal, fbal = np.eye(nX), V0, f
    rho = 1
    vol = 0
    for iter in range(max_iterations):
        # print('iteration  %s' % (iter))
        last_vol = vol

        # balance on every iteration (since V and Vdot are changing):
        if do_balance:
            [T, Vbal, fbal] = balance(x, V, f, S0 / rho, A)[0:3]
        else:
            T, Vbal, fbal = np.eye(nX), V, f
        # print('T is %s' % (T))
        V0bal = V0.Substitute(dict(zip(x, T@x)))
        # env = dict(zip(list(V0bal.GetVariables()), np.array([1, 2.31])))
        # print('V0bal is %s' % (V0bal.Evaluate(env)))

        [L1, sigma1] = findL1(x, fbal, Vbal, **kwargs)
        L2 = findL2(x, Vbal, V0bal, rho, **kwargs)
        [Vbal, rho] = optimizeV(x, fbal, L1, L2, V0bal, sigma1, **kwargs)

        vol = rho

        #  undo balancing (for the next iteration, or if i'm done)
        V = Vbal.Substitute(dict(zip(x, inv(T)@x)))
        if ((vol - last_vol) < converged_tol * last_vol):
            break
    print('iteration is %s' % iter)
    print('final rho is %s' % (rho))
    # env = dict(zip(x, np.array([1, 2.31])))
    # print('V is %s' % (V.Evaluate(env)))
    return V


def findL1(x, f, V, **kwargs):
    # print('finding L1')
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)
    degL1 = kwargs['degL1']
    # % construct multipliers for Vdot
    L1 = prog.NewFreePolynomial(Variables(x), degL1).ToExpression()

    # % construct Vdot
    Vdot = clean(V.Jacobian(x) @ f, x)
    # Vdot = V.Jacobian(x) @ f

    # env = dict(zip(x, np.array([1, 2.31])))
    # print('f0 is %s' % (f[0].Evaluate(env)))
    # print('f1 is %s' % (f[1].Evaluate(env)))
    # print('V is %s' % (V.Evaluate(env)))
    # % construct slack var
    sigma1 = prog.NewContinuousVariables(1, "s")[0]
    prog.AddConstraint(sigma1 >= 0)
    # % setup SOS constraints
    prog.AddSosConstraint(-Vdot + L1 * (V - 1) - sigma1 * V)
    prog.AddSosConstraint(L1)
    # add cost
    prog.AddCost(-sigma1)
    # result = Solve(prog)
    solver = MosekSolver()
    # solver.set_stream_logging(True, "")
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = solver.Solve(prog, None, None)
    print(result.get_solution_result())
    assert result.is_success()
    L1 = (result.GetSolution(L1))
    sigma1 = result.GetSolution(sigma1)
    # print('sigma1 is %s' % (sigma1))
    return L1, sigma1


def findL2(x, V, V0, rho, **kwargs):
    # print('finding L2')
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)
    degL2 = kwargs['degL2']
    # env = dict(zip(x, np.array([1, 2.31])))
    # print('V0 is %s' % (V0.Evaluate(env)))
    # print('V is %s' % (V.Evaluate(env)))
    # % construct multipliers for Vdot
    L2 = prog.NewFreePolynomial(Variables(x), degL2).ToExpression()
    # % construct slack var
    slack = prog.NewContinuousVariables(1, "s")[0]
    prog.AddConstraint(slack >= 0)
    # add normalizing constraint
    prog.AddSosConstraint(-(V - 1) + L2 * (V0 - rho))
    prog.AddSosConstraint(L2)
    prog.AddCost(slack)

    solver = MosekSolver()
    # solver.set_stream_logging(True, "")
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = solver.Solve(prog, None, None)
    # print(result.get_solution_result())
    assert result.is_success()
    L2 = (result.GetSolution(L2))
    # print(L2.Evaluate(env))
    return L2


def optimizeV(x, f, L1, L2, V0, sigma1, **kwargs):
    # print('finding V')
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)
    degV = kwargs['degV']
    # env = dict(zip(x, np.array([1, 2.31])))

    #% construct V
    V = prog.NewFreePolynomial(Variables(x), degV).ToExpression()

    # V = prog.NewSosPolynomial(Variables(x), degV)[0].ToExpression()
    Vdot = V.Jacobian(x) @ f
    # % construct rho
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)

    # % setup SOS constraints
    prog.AddSosConstraint(-Vdot + L1 * (V - 1) - sigma1 * V / 2)
    prog.AddSosConstraint(-(V - 1) + L2 * (V0 - rho))
    prog.AddSosConstraint(V)
    # % run SeDuMi/MOSEK and check output
    prog.AddCost(-rho)
    solver = MosekSolver()
    # solver.set_stream_logging(True, "")
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = solver.Solve(prog, None, None)
    # print(result.get_solution_result())
    assert result.is_success()
    V = result.GetSolution(V)
    # print(clean(V))
    rho = result.GetSolution(rho)
    # print('rho is %s' % (rho))
    return V, rho


def levelset_sos(system, V0, do_balance=False, write_to_file=False):
    x = system.sym_x
    f = system.sym_f
    nX = system.num_states
    prog = MathematicalProgram()
    prog.AddIndeterminates(x)
    xbar = x - system.x0
    if do_balance:
        [T, V, f, _, _] = balance(x, V0, f, None, None)
    else:
        T, V, f = np.eye(nX), V0, f
    Vdot = (V.Jacobian(x) @ f)

    # degV = system.degV
    degV = Polynomial(V, x).TotalDegree()
    degL1 = degV - 1 + system.degf
    degVdot = Polynomial(Vdot, x).TotalDegree()
    degxx = int(np.floor((degL1 + degVdot - degV) / 2))

    psi_deg = int(np.floor(max(2 * degxx + degV, degL1 + degVdot) / 2))
    f = lambda dim, deg: fact(dim + deg) // fact(dim) // fact(deg)
    psi_dim = f(nX, psi_deg)
    # print('equality-constrained SDP size is %s' % psi_dim)

    H = Jacobian(Vdot.Jacobian(x).T, x)
    env = dict(zip(x, system.x0))
    H = .5 * np.array([[i.Evaluate(env) for i in j]for j in H])
    print('eig of Hessian of Vdot %s' % (eig(H)[0]))
    assert (np.all(eig(H)[0] <= 0))
    # % construct slack var
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)
    L1 = prog.NewFreePolynomial(Variables(x), degL1).ToExpression()
    recast_cstr = 0
    if hasattr(system, 'recast'):
        L2 = prog.NewFreePolynomial(Variables(x), degL1).ToExpression()
        recast_cstr = L2 * system.recast  # system.recast contains the rigid body constraints for S-procedure
    prog.AddSosConstraint((x.T@x)**(degxx) * (V - rho) + L1 * Vdot + recast_cstr)

    prog.AddCost(-rho)
    solver = MosekSolver()

    # solver.set_stream_logging(True, log_file)
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = solver.Solve(prog, None, None)
    print(result.get_solution_result())
    # print('w/ solver %s' % (result.get_solver_id().name()))
    assert result.is_success()
    L1 = result.GetSolution(L1)
    rho = result.GetSolution(rho)
    # P = result.GetSolution(P)
    # print('cond # of gram is %s' % np.linalg.cond(P))
    print('rho is %s' % rho)
    V = V / rho
    V = (V.Substitute(dict(zip(x, inv(T) @ x))))
    return V


def convexly_search_for_V_on_samples(sampled_quantities, use_cvx=True,
                                     PSD_constrained=True,
                                     write_to_file=False):
    phi, eta = sampled_quantities
    monomial_dim = phi.shape[-1]
    num_samples = phi.shape[0]
    one_vec = np.ones(monomial_dim)
    constraints = []
    if use_cvx:
        P = cp.Variable((monomial_dim, monomial_dim), symmetric=True)
        slack = cp.Variable(1)
        if PSD_constrained:
            constraints += [P >> 0]
        constraints += [one_vec.T@P@one_vec == 1]

        for i in range(num_samples):
            this_vdot = phi[i, :].T@P@eta[i, :]
            constraints += [this_vdot <= slack]

        prob = cp.Problem(cp.Minimize(slack), constraints)
        prob.solve(solver=cp.MOSEK, verbose=True)
        P = P.value
        slack = slack.value

    else:
        prog = MathematicalProgram()
        P = prog.NewSymmetricContinuousVariables(monomial_dim, "P")
        slack = prog.NewContinuousVariables(1, "s")[0]
        if PSD_constrained:
            prog.AddPositiveSemidefiniteConstraint(P)
        for i in range(num_samples):
            if not PSD_constrained:
                # reduces to an LP
                this_v = phi[i, :].T@P@phi[i, :]
                prog.AddConstraint(this_v >= 0)
            this_vdot = phi[i, :].T@P@phi[i, :].T@P@eta[i, :]
            prog.AddConstraint(this_vdot <= slack)

        prog.AddConstraint(one_vec.T@P@one_vec == 1)
        prog.AddCost(slack)
        solver = MosekSolver()
        if write_to_file:
            log_file = "convexly_search_for_V_on_samples.txt"
        else:
            log_file = ""
        # solver.set_stream_logging(True, log_file)
        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        prog.SetSolverOptions(options)
        result = solver.Solve(prog, None, None)
        print(result.get_solution_result())
        assert result.is_success()
        P = result.GetSolution(P)
        slack = result.GetSolution(slack)

    print('eig of orignal P  %s' % (eig(P)[0]))
    print('slack value %s' % slack)
    return P


def balance_quad_form(S, P):
    # copied from the old drake, with only syntax swap
    #  Quadratic Form "Balancing"
    #
    #    T = balqf(S,P)
    #
    #  Input:
    #    S -- n-by-n symmetric positive definite.
    #    P -- n-by-n symmetric, full rank.
    #
    #  Finds a T such that:
    #    T'*S*T = D
    #    T'*P*T = D^(-1)

    # if np.linalg.norm(S - S.T, 1) > 1e-8:
        # raise Error('S must be symmetric')
    # if np.linalg.norm(P - P.T, 1) > 1e-8:
        # raise Error('P must be symmetric')
    # if np.linalg.cond(P) > 1e10:
        # raise Error('P must be full rank')

    # Tests if S positive def. for us.
    V = inv(np.linalg.cholesky(S).T)
    [N, l, U] = np.linalg.svd((V.T.dot(P)).dot(V))
    if N.ravel()[0] < 0:
        N = -N
    T = (V.dot(N)).dot(np.diag(np.power(l, -.25, dtype=float)))
    D = np.diag(np.power(l, -.5, dtype=float))
    return T, D


def balance(x, V, f, S, A):
    if S is None:
        H = Jacobian(V.Jacobian(x).T, x)
        env = dict(zip(x, np.zeros(x.shape)))
        S = .5 * np.array([[i.Evaluate(env) for i in j]for j in H])
    if A is None:
        J = Jacobian(f, x)
        env = dict(zip(x, np.zeros(x.shape)))
        A = np.array([[i.Evaluate(env) for i in j]for j in J])
    [T, D] = balance_quad_form(S, (S@A + A.T@S))
    # print('T is %s' % (T))
    # Sbal = (T.T)@(S)@(T)
    Vbal = V.Substitute(dict(zip(x, T@x)))
    fbal = inv(T)@[i.Substitute(dict(zip(x, T@x))) for i in f]
    return T, Vbal, fbal, S, A


def clean(poly, x, tol=1e-9):
    if isinstance(poly, Expression):
        poly = Polynomial(poly, x)
    return poly.RemoveTermsWithSmallCoefficients(tol).ToExpression()


################
# class opt:

#     def __init__(self, nX, degf, degV=4, converged_tol=.01, max_iterations=10,
#                  degL1=None, degL2=None, do_balance=False):
#         self.nX = nX
#         self.degf = degf
#         self.degV = degV
#         self.converged_tol = converged_tol
#         self.max_iterations = max_iterations
#         if degL1 is None:
#             degL1 = degV - 1 + degf
#         self.degL1 = degL1
#         if degL2 is None:
#             degL2 = degL1
#         self.degL2 = degL2
#         self.do_balance = do_balance


def levelset_w_feature_transformation(system, gram, g, L1):
    # dirty, hard coded everything
    f = system.sym_f
    x = system.sym_x
    V = Polynomial(system.sym_V, x)
    Vdot = Polynomial(system.sym_Vdot, x)
    V = system.sym_V
    Vdot = system.sym_Vdot
    # xxd = Polynomial(system.sym_xxd,x)

    prog = MathematicalProgram()
    prog.AddIndeterminates(x)

    # % construct slack var
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)

    basis = MonomialBasis(x, 8)
    candidate = Polynomial()
    P = prog.NewSymmetricContinuousVariables(45, "P")
    prog.AddPositiveSemidefiniteConstraint(P)

    for i in range(45):
        candidate.AddProduct((P[i, i]), basis[i] * basis[i])
        for j in range(i + 1, 45):
            candidate.AddProduct(2 * (P[i, j]), basis[i] * basis[j])

    dim_psi = system.sym_psi.shape[0]
    L1 = prog.NewFreePolynomial(Variables(x), 8).ToExpression()
    P = prog.NewSymmetricContinuousVariables(dim_psi, "P")
    prog.AddPositiveSemidefiniteConstraint(P)

    # newbasis = np.array([Polynomial(i,x) for i in system.sym_psi.T@g])
    candidate = Polynomial()
    trans_P = g@P@g.T
    # basis =system.sym_psi
    basis = MonomialBasis(x, 8)

    for i in range(dim_psi):
        candidate.AddProduct((trans_P[i, i]), basis[i] * basis[i])
        for j in range(i + 1, dim_psi):
            candidate.AddProduct(2 * (trans_P[i, j]), basis[i] * basis[j])

    # candidateDecomp = system.sym_psi.T@trans_P@system.sym_psi
    # residual = candidateDecomp-candidate.ToExpression()
    levelsetPoly = Polynomial((x.T@x)**(5) * (V - rho) + L1 * Vdot, x)
    # levelsetPoly = (xxd * (V*Polynomial(rho,x)-1) + L1 *(Vdot))
    prog.AddEqualityConstraintBetweenPolynomials(candidate, levelsetPoly)

    prog.AddCost(-rho)
    solver = MosekSolver()
    # solver.set_stream_logging(True, "")
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    result = solver.Solve(prog, None, None)
    print(result.get_solution_result())
    # print('w/ solver %s' % (result.get_solver_id().name()))
    assert result.is_success()
    # L1 = result.GetSolution(L1)
    rho = result.GetSolution(rho)
    print(rho)
    V = V.ToExpression() * rho
    return V


def check_residual(system, gram, rho, L, x_val):
    f = system.sym_f
    x = system.sym_x
    V = system.sym_V
    Vdot = system.sym_Vdot
    l_coeffs = L.T
    L1 = l_coeffs@system.sym_sigma
    candidateDecomp = system.sym_psi.T@gram@system.sym_psi
    levelsetPoly = (system.sym_xxd * (V * rho - 1) + L1 * Vdot)[0]
    env = dict(zip(x, x_val.T))
    ratio = levelsetPoly.Evaluate(env) / candidateDecomp.Evaluate(env)
    print(ratio)
    residual = Polynomial((levelsetPoly - candidateDecomp), x)
    residual_coeffs_mapping = residual.monomial_to_coefficient_map()
    coeffs = list(residual_coeffs_mapping.values())
    for i in coeffs:
        print(i)


def recast_poly_back_to_nonlinear(V, CL_sys):
    """Recast the polynomial Lyapunov candidate back to the original coornidate.
    e.g. originally, we might have recast the non-linearity sin(x) as s, so
    that we have a V=x**2+s**2. In order to visualize, we need the inverse
    mapping, so that the levelset of V can be properly plotted in the 'x'
    cooridnate

    Args:
        x (Inderterminates):
        V (Inderterminates): V(x)
        CL_sys(ClosedLoopSystem): encodes the mapping
    """
    env = CL_sys.inverse_recast_map()
    nonPolyV = V.Substitute(env)
    return nonPolyV


def IQC_tanh(x, y):
    y_cross = 0.642614
    x_off = .12
    return np.hstack((y ** 2 - 1,
                      (y - ((x) + x_off)) * (y + y_cross),
                      (y - ((x) - x_off)) * (y - y_cross),
                      (y - x) * y))


# def levelsetLP(system, gram):
#     f = system.sym_f
#     x = system.sym_x
#     V = system.sym_V
#     Vdot = system.sym_Vdot
#     prog = MathematicalProgram()

#     # % construct slack var
#     rho = prog.NewContinuousVariables(1, "r")[0]
#     prog.AddConstraint(rho >= 0)
#     scaling = prog.NewContinuousVariables(gram.shape[0], "s")
#     for i in scaling:
#         prog.AddConstraint(i >= 0)
#     slack = prog.NewContinuousVariables(1, "l")[0]

#     l_coeffs = prog.NewContinuousVariables(system.sym_sigma.shape[0], "L")
#     L1 = l_coeffs@system.sym_sigma
# candidateDecomp =
# Polynomial(system.sym_psi.T@np.diag(scaling)@gram@system.sym_psi, x)

#     levelsetPoly = Polynomial(
#         system.sym_xxd * (V - rho) + L1 * Vdot + slack, x)
#     prog.AddEqualityConstraintBetweenPolynomials(candidateDecomp, levelsetPoly)
#     prog.AddCost(-rho)
#     solver = MosekSolver()
#     solver.set_stream_logging(True, "")
#     result = solver.Solve(prog, None, None)
#     print(result.get_solution_result())
#     # print('w/ solver %s' % (result.get_solver_id().name()))
#     assert result.is_success()
#     L1 = result.GetSolution(L1)
#     rho = result.GetSolution(rho)
#     print(rho)
#     V = V / rho
#     V = (V.Substitute(dict(zip(x, inv(T) @ x))))
#     return V
