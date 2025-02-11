# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import functools as ft
import scipy as sp


def purity(r):
    """
    Purity of the state r, calculated as Tr(r**2).
    :param r: Matrix
    :return: Real number
    """
    return np.trace(np.matmul(r, r))


def random_pure(dim):
    """
    A pure state/projector of dimension dim.
    :param dim:
    :return: Column Matrix
    """
    proj = np.transpose([np.array(np.random.normal(0, 1, dim) + complex(0, 1) * np.random.normal(0, 1, dim))])
    proj /= np.linalg.norm(proj)
    return proj


def random_pure_dl(dim_list):
    """
    Generateds a random pure state (ket) with subsystem dimensions from dim_list
    :param dim_list: List of integers
    :return: Column Matrix
    """
    ket_list = [random_pure(d) for d in dim_list]
    return ft.reduce(np.kron, ket_list)


def matrix_ejk(aj, ak, dim):
    """
    A dim x dim matrix with a 1 at index (aj,ak), and zeroes elsewhere.
    :param aj: int
    :param ak: int
    :param dim: int
    :return: Matrix
    """
    mat = np.zeros((dim, dim), complex)
    mat[aj, ak] = complex(1, 0)
    return mat


def gell_mann_basis(dim: int):
    """
    A list of (dim**2 -1) matrices that are the generators of SU(dim) Lie Algebra
    :param dim:
    :return: A list of matrices
    """
    idm = np.identity(dim, complex)
    idm /= np.linalg.norm(idm)
    # symmetric matrices in the basis:
    sym = np.zeros((int((dim * (dim - 1) / 2)), dim, dim), complex)
    idx = 0
    for k in range(dim):
        for j in range(k):
            sym[idx] = matrix_ejk(j, k, dim) + matrix_ejk(k, j, dim)
            sym[idx] /= np.linalg.norm(sym[idx])
            idx += 1
    # antisymmetric matrices in the basis
    anti_sym = np.zeros((int((dim * (dim - 1) / 2)), dim, dim), complex)
    idx = 0
    for k in range(dim):
        for j in range(k):
            anti_sym[idx] = complex(0, -1) * (matrix_ejk(j, k, dim) - matrix_ejk(k, j, dim))
            anti_sym[idx] /= np.linalg.norm(anti_sym[idx])
            idx += 1
    # diagonal matrices in the basis
    diag = np.zeros((dim - 1, dim, dim), complex)
    idx = 0
    for k in range(dim - 1):
        for j in range(k + 1):
            diag[idx] += matrix_ejk(j, j, dim)
        diag[idx] -= (k + 1) * matrix_ejk(k + 1, k + 1, dim)
        diag[idx] *= complex(np.sqrt(2 / ((k + 1) * (k + 2))), 0)
        diag[idx] /= np.linalg.norm(diag[idx])
        idx += 1
    return np.concatenate((sym, anti_sym, diag, [idm]))


def hs_distance(r1, r2):
    """
    Squared Hilbert-Schmidt distance between r1 and r2
    :param r1: Matrix
    :param r2: Matrix
    :return: distance as real number
    """
    return (np.linalg.norm(r1 - r2)).real ** 2


def pre_sel(r0, r1, r2):
    return np.trace((r0 - r1) @ (r2 - r1)).real

# def to_maximize(x, *args):
# Old definition with more expm operations
#     rho2, rho3, base_un, dim_list, lenlist, herm = args
#     # herm = [np.eye(d) for d in dim_list]
#     # lenlist = [len(base_un[i]) for i in range(len(base_un))]
#     offset = 0
#     l_base = len(base_un)
#     for i in range(l_base):
#         herm[i] = sum(1j * x[j + offset] * base_un[i][j] for j in range(lenlist[i]))
#         herm[i] = sp.linalg.expm(herm[i])
#         offset += lenlist[i]
#     unit = ft.reduce(np.kron, herm)
#     rho2u = unit @ (rho2 @ np.transpose(np.conjugate(unit)))
#     # print(unit)
#     return -np.trace(rho3 @ rho2u).real

def to_maximize(x, *args):
    # New definition with less expm operations
    rho2, rho3, base_un, dim_list, lenlist, herm = args
    # herm = [np.eye(d) for d in dim_list]
    # lenlist = [len(base_un[i]) for i in range(len(base_un))]
    offset = 0
    l_base = len(base_un)
    exponent = np.zeros((np.prod(dim_list), np.prod(dim_list)), complex)
    for i in range(l_base):
        herm[i] = sum(1j * x[j + offset] * base_un[i][j] for j in range(lenlist[i]))
        list_id = np.array([np.eye(d, dtype=complex) for d in dim_list])
        list_id[i] = herm[i]
        exponent += ft.reduce(np.kron, list_id)
        offset += lenlist[i]
    unit = sp.linalg.expm(exponent)
    rho2u = unit @ (rho2 @ np.transpose(np.conjugate(unit)))
    # print(unit)
    return -np.trace(rho3 @ rho2u).real


def optimize_rho2(rho0, rho1, rho2_ket, pre1, nq, dim_list, basis_unitary):
    rho3 = rho0 - rho1
    rho2 = make_density(rho2_ket)
    herm = [np.eye(d, dtype=complex) for d in dim_list]
    lenlist = dim_list*dim_list
    x0 = np.ones(sum(lenlist))
    x_bounds = sp.optimize.Bounds(np.zeros_like(x0), np.full_like(x0, 10), True)
    # res = sp.optimize.minimize(to_maximize, x0, (rho2, rho3, basis_unitary, dim_list, lenlist, herm),
    # method='Powell', bounds=x_bounds)
    # res = sp.optimize.basinhopping(to_maximize, x0, 100,
    #                                minimizer_kwargs={'args': (rho2, rho3, basis_unitary, dim_list, lenlist, herm)})
    res = sp.optimize.differential_evolution(to_maximize, x_bounds,(rho2, rho3, basis_unitary, dim_list, lenlist, herm))
    #res = sp.optimize.minimize(to_maximize, x0,
    #                           (rho2, rho3, basis_unitary, dim_list, lenlist, herm), 'Nelder-Mead')
    # res = sp.optimize.dual_annealing(to_maximize, x_bounds,
    #                                        (rho2, rho3, basis_unitary, dim_list, lenlist, herm))
    # minimizer_kwargs={args:(rho2, rho3, basis_unitary, dim_list, lenlist, herm)})
    xres = res.x
    offset = 0
    l_base = len(basis_unitary)
    exponent = np.zeros((np.prod(dim_list), np.prod(dim_list)), complex)
    for i in range(l_base):
        herm[i] = sum(1j * xres[j + offset] * basis_unitary[i][j] for j in range(lenlist[i]))
        list_id = np.array([np.eye(d, dtype=complex) for d in dim_list])
        list_id[i] = herm[i]
        exponent += ft.reduce(np.kron, list_id)
        offset += lenlist[i]
    unit = sp.linalg.expm(exponent)
    rho4 = unit @ rho2 @ np.transpose(np.conjugate(unit))
    pre2 = pre_sel(rho0, rho1, rho4)
    if pre2 > pre1:
        return pre2, rho4
    else:
        return pre1, rho2


def gilbert(rho_in: np.array, nq: int, dim_list: np.array, max_iter: int, max_trials: int, opt_state="on", rng_seed=666, rho1_in=None):
    """
    Approximate the Closest Separable State (CSS) to the given state rho_in using Gilbert's algorithm
    :param rho_in: matrix
    :param nq: number of subsystems (int)
    :param dim_list: list of subsystem dimensions (int)
    :param max_iter: max iterations/corrections
    :param max_trials: maximum trials
    :param rng_seed: seed for RNG
    :param opt_state: Optimization On/Off, default "on"
    :param rho1_in: Optional start state
    :return: CSS, min HS-distance, number of trials
    """
    np.random.seed(rng_seed)
    rho0 = rho_in
    if rho1_in is None:
        rho1 = np.diag(np.diag(rho))
    else:
        rho1 = rho1_in
    # ndim = np.multiply.reduce(dim_list)
    print(rho1)
    rho2_ket = random_pure_dl(dim_list)
    rho2 = make_density(rho2_ket)
    dist0 = hs_distance(rho0, rho1)
    trials = 1
    basis_unitary = [gell_mann_basis(x) for x in dim_list]
    pre1 = pre_sel(rho0, rho1, rho2)
    pre2 = 0.
    iter = 1
    while iter <= max_iter and trials <= max_trials:
        # if iter % 5 == 0:
        #       print(iter, dist0)
        print(iter, dist0)

        while pre1 < 0 and trials <= max_trials:
            rho2_ket = random_pure_dl(dim_list)
            rho2 = make_density(rho2_ket)
            pre1 = pre_sel(rho0, rho1, rho2)
            trials += 1
        if trials > max_trials:
            break
        if opt_state == "on" or opt_state == "On":
            pre1, rho2 = optimize_rho2(rho0, rho1, rho2_ket, pre1, nq, dim_list, basis_unitary)
        p = 1 - pre1 / hs_distance(rho1, rho2)
        dist1 = hs_distance(rho0, p * rho1 + (1 - p) * rho2)
        if 0 <= p <= 1 and dist1 < dist0:
            iter += 1
            rho1 = p * rho1 + (1 - p) * rho2
            dist0 = dist1
        rho2_ket = random_pure_dl(dim_list)
        rho2 = make_density(rho2_ket)
        pre1 = pre_sel(rho0, rho1, rho2)
        trials += 1
    return rho1, dist0, trials


def make_density(ket):
    return ket @ np.transpose(np.conjugate(ket))


def get_diagonal(rho):
    return np.diag(np.diag(rho))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # rho = random_pure_dl([2, 2])
    # rho = make_density(rho)
    dim_list = np.array([2, 2, 2])
    rho = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]).astype(complex)
    # rho = sp.io.mmread("wxstate.mtx")
    print(rho)
    # approx = sp.io.mmread("wxcss.mtx")
    css, dist, trials = gilbert(rho, 3, dim_list, 10, 10000000000, opt_state="on")  # , rho1_in=approx)

    print(css, dist, trials)
    sp.io.mmwrite("wxcss.mtx", css)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
