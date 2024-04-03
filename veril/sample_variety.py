from pydrake.all import (Polynomial, Variable, Evaluate, Substitute,
                         SolverOptions, CommonSolverOption,
                         MathematicalProgram, MosekSolver)
import numpy as np
from numpy.linalg import eig, inv
import time
from veril.util.plots import *
import math
import os

"""Summary
x: sampled state space or indeterminates
Y = [x, V, xxd, trans_psi, T] are all that's necessary for the SDP
T: transformation matrix due to the coordiante ring, serves to reduce the
dimension of the monomial_basis down
P: r.h.s. SOS decomposition Gram matrix
rho: scalar Lyapunov level
"""


def verify_via_variety(system, V, init_root_threads=1, num_samples=10):
    assert system.loop_closed
    if V is None:
        A, S, V = system.linearized_quadractic_V()
    system.set_sample_variety_features(V)
    Vdot = V.Jacobian(system.sym_x) @ system.sym_f
    # Vdot = system.sym_Vdot
    variety = multi_to_univariate(Vdot)
    is_vanishing = False
    filepath = '../data/' + system.name + '/variety_Y.npz'
    rho_max =0
    for i in range(num_samples):
        if os.path.exists(filepath+'no'):
            ff = np.load(filepath)
            Y = [ff['x'], ff['V'], ff['xxd'], ff['psi'], ff['T']]
        else:
            x = sample_on_variety(variety, init_root_threads, system.x0)
            Y = x_to_Y(system, x, variety)
        while not is_vanishing:
            start = time.time()
            sym_V, rho, P, Y = solve_SDP_on_samples(system, Y)
            is_vanishing, test_x = check_vanishing(system, variety, rho, P, Y)
            end = time.time()
            print('sampling variety time %s' % (end - start))
            test_Y = x_to_Y(system, test_x, variety, test_only=True)
            try:
                Y = [np.concatenate([i, j])
                     for (i, j) in zip(Y[:-1], test_Y[:-1])] + [Y[-1]]
            except:
                print("Y concatenation error")
            rho_max = rho if rho>rho_max else rho_max
        print(f"rho: {rho_max}")
        plot_funnel(V / rho_max, system, slice_idx=system.slice,
                add_title=' - Sampling Variety ' + 'Result')
    return V, rho


def multi_to_univariate(variety):
    """helper function, turn a multivariate polynomial in x into univariate
    poly by x = a*t+b transformation. For now, a,b are both variables, but will
    be substituded w/ random numbers for roots finidng.

    this is faster than doing Polynomial coefficient mapping under the current
    pydrake class and API design

    Args:
        variety (TYPE): pydrake scalar Expression

    Returns:
        TYPE: List of all necessary parameters to describe the univariate
        polynomial to sampel from
    """
    x = np.array(list(variety.GetVariables()))
    nx = x.shape[0]
    t = Variable('t')
    prog = MathematicalProgram()
    a = prog.NewIndeterminates(nx, "a")
    b = prog.NewIndeterminates(nx, "b")
    x_in_t = a * t + b

    # start = time.time()
    variety_poly = Polynomial(variety, x)
    key_val_map = variety_poly.monomial_to_coefficient_map()
    key_in_x = [i.ToExpression() for i in list(key_val_map.keys())]
    coeffs_in_x = [i.Evaluate({t: 0}) for i in list(key_val_map.values())]
    key_in_t = [i.Substitute(dict(zip(x, x_in_t))) for i in key_in_x]
    basis_in_t = [Polynomial(i) for i in key_in_t]
    end = time.time()
    # print(end-start)

    # start = time.time()
    # v_in_tab = Polynomial(variety.Substitute(dict(zip(x,x_in_t))))
    # end = time.time()
    # print(end-start)
    return [variety_poly, basis_in_t, coeffs_in_x, a, b, t, x, nx]


def sample_on_variety(variety, root_threads, x0, slice_idx=None):
    """returns the pure 'x' samples on one given variety. i.e. return the roots
    to the given multi-variate polynomial. Note that, due to the numerical
    accuracy of numpy's polyroots, only returning all the real roots.

    Args:
        variety (list)
        root_threads (int): number of 'attempts' of turning the multi-variate
        slice_idx (None, optional): Description
        into univarite, i.e., the final num_roots != root_threads. (since
        there're d complex roots for a univariate polynomial of degree d, and
        thus more than 1 real root)

    Returns:
        n_roots-by-num_states (num_x in the variety): all real roots for the
        univarite polynomial coming from the root_threads attemps of
        transformation
    """
    [variety_poly, basis_in_t, coeffs_in_x, a, b, t, x, nx] = variety
    samples = np.zeros((nx,))
    num_roots = 0

    while num_roots < root_threads:
        # start = time.time()
        if slice_idx is None:
            alphas = np.random.randn(nx)
            betas = np.random.randn(nx)
            # betas = np.zeros((nx,))
        else:
            alphas = np.zeros((nx,))
            betas = np.zeros((nx,))
            alphas[slice_idx[0]] = np.random.randn(1)
            alphas[slice_idx[1]] = np.random.randn(1)
            betas[slice_idx[0]] = np.random.randn(1)
            betas[slice_idx[1]] = np.random.randn(1)

        env = dict(zip(a, alphas))
        env.update(dict(zip(b, betas)))
        this_basis = np.array([i.EvaluatePartial(env) for i in basis_in_t])
        Vt = this_basis@coeffs_in_x
        coeffs = [i.Evaluate({t: 0}) for i in list(
            Vt.monomial_to_coefficient_map().values())]
        root_t = np.polynomial.polynomial.polyroots(coeffs)
        # choose to take only the real roots
        root_t = root_t[~np.iscomplex(root_t)].real
        # end = time.time()
        # print(end - start)
        if len(root_t) == 0:
            continue
        root_x = np.array([alphas * i + betas for i in root_t])
        root_x = root_x[~np.all(root_x == x0, axis=1)]
        root_x = root_x[~np.any(np.isclose(root_x, 0, atol=1e-1), axis=1)]
        root_x = root_x[~np.any(np.abs(root_x) > 3.5, axis=1)]

        # plug in back the t value to evaluate the polynomial value
        varitey_x = [variety_poly.Evaluate(dict(zip(x, i))) for i in
                     root_x]
        # double check the evaluation is close to zero
        close = np.isclose(varitey_x, 0, atol=4e-12)

        if np.any(close):
            root_x = root_x[close]
            samples = np.vstack([samples, root_x])
            num_roots = samples.shape[0] - 1
    return samples[1:]


def x_to_Y(system, x, variety, do_transform=True, test_only=False):
    enough_x = False
    first = False
    while not enough_x:
        for i in (range(500) if first == True else range(1)):
            more_x = sample_on_variety(variety, 1, system.x0)
            x = np.vstack([x, more_x])
            V = system.get_v_values(x)
            x, V = balancing_V(x, V)
            print(i)
        first=False
        [xxd, psi] = system.get_sample_variety_features(x)
        trans_psi, T = coordinate_ring_transform(psi, do_transform, test_only)
        enough_x = check_genericity(trans_psi, test_only)

    scale = min(V)
    for i in range(V.shape[0]):
        xxd[i] = xxd[i] / scale
        trans_psi[i] = trans_psi[i] / np.sqrt(scale)
    Y = [x, V, xxd, trans_psi, T]
    if not test_only:
        path = '../data/' + system.name + '/variety_Y.npz'
        # np.savez_compressed(path, x=Y[0], V=Y[1], xxd=Y[2], psi=Y[3], T=Y[4])
    return Y


def coordinate_ring_transform(psi, do_transform, test_only):
    """reduce the dimensionality of the sampled-monimials by taking advantage
    of the coordiate ring structure (similar to Gaussian elimination used in
    finding Grobner basis)

    Args:
        psi: (num_samples, monomial_dim)
        U =psi.T (monomial_dim, num_samples)
        [u,s,v] = svd(U)
        n = # of non-zero values in s
        U= T@U_transformed, where
        for testing, standard_monomial = T * reduced_basis, or
        pinv(T)@standard_monomial = reduced_basis

    Returns:
        transformed_basis (num_samples, reduced_monomials)
        T (reduced_monomials, monomial_dim)
    """
    if test_only or not do_transform:
        return psi, np.eye(psi.shape[1])
    U = psi.T
    [u, diag_s, v] = np.linalg.svd(U)
    tol = max(U.shape) * diag_s[0] * 1e-16
    original_monomial_dim, num_samples = U.shape
    n = sum(diag_s > tol)
    print('original_monomial_dim is %s' % original_monomial_dim)
    print('rank for SVD %s' % n)
    print('coordinate_ring_transform for %s samples' % num_samples)
    if n / original_monomial_dim >= .95:
        # print('no need for transformation')
        return U.T, np.eye(original_monomial_dim)
    else:
        print('does transforming')
        s = np.zeros(U.shape)
        np.fill_diagonal(s, diag_s)
        U_transformed = v[:n, :]
        T = u@s[:, :n]
        T = np.linalg.pinv(T)
        transformed_basis = U_transformed.T
        assert np.allclose(T@U, U_transformed)
        return transformed_basis, T

# U=np.array([[1,1,1,0,0,0],
# [-0.6, -1.2, -0.75, 0.8,0.4,0.25],
# [-1.2,-.6,-.75,-.4,-.8,-.25],
# [1.2,0.6,.75,.4,.8,.25],
# [-.6,-1.2,-.75,.8,.4,.25]])
# coordinate_ring_transform(U.T)


def check_genericity(psi, test_only):
    """check the rank condition of the samples to make sure the genericity is
    satisfied

    Args:
        psi (ndarry): (m,n), current samples,

    Returns:
        enough_samples (Bool): if the current sample set is generic enough
        m0 (int): current samples rank, gives good indicator of whether to
        augment or truncate the current sample set

    Deleted Parameters:
        m (int): current number of samples
        n (int): monomial_dim (or reduced_monomial if do_transformation)
    """
    if test_only:
        return True
    enough_samples = True
    m, n = psi.shape
    n2 = n * (n + 1) / 2
    m0 = min(m, n2)
    # sub_samples = psi[:m0, :]
    sub_samples = psi

    c = np.power(sub_samples@sub_samples.T, 2)  # c = q'*q
    # print('c shape is %s' % str(c.shape))
    s = abs(np.linalg.eig(c)[0])
    tol = max(c.shape) * np.spacing(max(s)) * 1e3
    sample_rank = sum(s > tol)
    # print('sample rank is %s' % sample_rank)
    if sample_rank == m0 and sample_rank < n2:
        # meaning m<n2 and sample full rank
        # print('Insufficient samples!!')
        enough_samples = False
    return enough_samples


def solve_SDP_on_samples(system, Y, write_to_file=False):
    prog = MathematicalProgram()
    rho = prog.NewContinuousVariables(1, "r")[0]
    prog.AddConstraint(rho >= 0)

    [x, V, xxd, psi, T] = Y
    # print('SDP V %s' % V)
    num_samples, dim_psi = psi.shape
    # print('num_samples is %s' % num_samples)
    print('dim_psi is %s' % dim_psi)
    P = prog.NewSymmetricContinuousVariables(dim_psi, "P")
    prog.AddPositiveSemidefiniteConstraint(P)

    for i in range(psi.shape[0]):
        residual = xxd[i] * (V[i] - rho) - psi[i].T@P@psi[i]
        prog.AddConstraint(residual == 0)

    prog.AddCost(-rho)
    solver = MosekSolver()
    options = SolverOptions()
    options.SetOption(CommonSolverOption.kPrintToConsole, 1)
    prog.SetSolverOptions(options)
    log_file = "sampling_variety_SDP.text" if write_to_file else ""
    # solver.set_stream_logging(True, log_file)
    result = solver.Solve(prog, None, None)
    # print(result.get_solution_result())
    assert result.is_success()
    P = result.GetSolution(P)
    rho = result.GetSolution(rho)
    # V = system.sym_V / rho
    # TODO: double check if scaling at this stage affects the downstream
    # oversampling
    print(rho)
    return system.sym_V, rho, P, Y


def check_vanishing(system, variety, rho, P, Y):
    T = Y[-1]
    empty_test = True
    while empty_test:
        test_x = sample_on_variety(variety, 1, system.x0)
        V = system.get_v_values(test_x)
        old_V = Y[1]
        idx = old_new_V(old_V, V)
        empty_test = (len(idx) == test_x.shape[0])
        test_x = np.delete(test_x, idx, axis=0)
        V = np.delete(V, idx, axis=0)

    [xxd, psi] = system.get_sample_variety_features(test_x)
    isVanishing = True
    # print('vanishing V %s' % V)
    idx = []
    for i in range(test_x.shape[0]):
        levelset = xxd[i] * (V[i] - rho)
        this_psi = T@psi[i]
        candidate = this_psi.T@P@this_psi
        # print('levelset %s' %levelset)
        # print('candidate %s' %candidate)
        isclose = math.isclose(levelset, candidate, rel_tol=1e-2, abs_tol=1e-2)
        ratio = (levelset - candidate) / candidate
        print('two polynomials evals ratio %s' % ratio)
        # if ratio < .97 or ratio > 1.1:
        if not isclose:
            isVanishing = False
            idx += [i]
    # scatterSamples(test_x, system)
    return isVanishing, test_x[idx]


def balancing_V(x, V, tol=1e3):
    balanced = max(V) / min(V) < tol
    while not balanced:
        print('not balanced')
        idx = [np.argmax(V), np.argmin(V)]
        # test_x = np.vstack([test_x, x[idx]])
        x = np.delete(x, idx, axis=0)
        V = np.delete(V, idx, axis=0)
        balanced = max(V) / min(V) < tol
    return x, V


def old_new_V(oldv, newv):
    idx = []
    for i in range(newv.shape[0]):
        if newv[i] / max(oldv) > 10 or newv[i] / min(oldv) < 1e-2:
            print('unbalanced')
            idx += [i]
    return idx
