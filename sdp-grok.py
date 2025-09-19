import numpy as np
import cvxpy as cp

# Partial transpose matrix
def get_partial_transpose_matrix():
    T = np.zeros((16, 16))
    basis = [(i, j) for i in range(2) for j in range(2)]
    for k, (i, j) in enumerate(basis):
        l = 2 * j + i
        T[l, k] = 1
    return T

def apply_partial_transpose(sigma):
    T = get_partial_transpose_matrix()
    sigma_vec = cp.vec(sigma)
    sigma_pt_vec = T @ sigma_vec
    return cp.reshape(sigma_pt_vec, (4, 4))

# Bell state
rho = np.array([[0.5, 0, 0, 0.5],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.5, 0, 0, 0.5]], dtype=complex)

# Verify rho
assert np.allclose(rho, rho.conj().T), "rho must be Hermitian"
assert np.isclose(np.trace(rho), 1), "Trace of rho must be 1"
assert np.all(np.linalg.eigvals(rho) >= -1e-10), "rho must be PSD"

# Dimensions
n = 4

# SDP variables
t = cp.Variable()
sigma = cp.Variable((n, n), hermitian=True)

# Identity
I = np.eye(n)

# Distance constraint
diff = rho - sigma
diff_vec = cp.vec(diff)
distance_constraint = cp.bmat([[t, diff_vec.H],
                              [diff_vec, I]])

# Constraints
constraints = [
    distance_constraint >> 0,
    sigma >> 0,
    cp.trace(sigma) == 1,
    apply_partial_transpose(sigma) >> 0
]

# Objective
objective = cp.Minimize(t)

# Solve with higher precision
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS, eps=1e-10, verbose=True)

# Results
optimal_t = None
optimal_sigma = None

if problem.status == cp.OPTIMAL:
    optimal_t = t.value
    optimal_sigma = sigma.value
    print("Optimal squared Hilbert-Schmidt distance:", optimal_t)
    print("Optimal separable state sigma:\n", np.round(optimal_sigma.real, 6))
    print("Trace of sigma:", np.trace(optimal_sigma).real)
    print("Is sigma Hermitian?", np.allclose(optimal_sigma, optimal_sigma.conj().T))
    sigma_pt_val = apply_partial_transpose(optimal_sigma).value
    print("Is sigma PPT?", np.all(np.linalg.eigvals(sigma_pt_val) >= -1e-10))
    diff = rho - optimal_sigma
    computed_distance = np.trace(diff.conj().T @ diff).real
    print("Computed squared distance:", computed_distance)
else:
    print("Problem status:", problem.status)

# Eigenvalues
if optimal_sigma is not None:
    eigvals_sigma = np.linalg.eigvals(optimal_sigma)
    print("Eigenvalues of sigma:", np.round(eigvals_sigma.real, 6))

# Validate against known solution
sigma_theory = (1/3) * rho + (2/3) * (np.eye(4) / 4)
diff_theory = rho - sigma_theory
distance_theory = np.trace(diff_theory.conj().T @ diff_theory).real
print("\nTheoretical closest separable state:\n", np.round(sigma_theory.real, 6))
print("Theoretical squared distance:", distance_theory)