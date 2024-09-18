import numpy as np

def interior_point_method(f, grad_f, hessian_f, solve_kkt_system, A, b, G, h, x0, s0, z0, 
                          tol=1e-6, max_iter=100):
  """
  Minimizes a convex function f(x) subject to linear equality and inequality constraints
  using a primal-dual interior point method with Nesterov-Todd scaling and Mehrotra correction.

  Args:
    f: The convex function to minimize.
    grad_f: Function to compute the gradient of f.
    hessian_f: Function to compute the Hessian of f.
    solve_kkt_system: Function to solve the KKT system.
                          Signature: delta_x, delta_z, delta_s = solve_kkt_system(H, A, G, W, r_d, r_p, r_g)
    A: The equality constraint matrix (A @ x = b).
    b: The equality constraint vector.
    G: The inequality constraint matrix (G @ x <= h).
    h: The inequality constraint vector.
    x0: Initial guess for the primal variable x.
    s0: Initial values for the slack variables s in the inequality constraints (s > 0).
    z0: Initial values for the dual variables z corresponding to the inequality constraints (z > 0).
    tol: Tolerance for convergence.
    max_iter: Maximum number of iterations.

  Returns:
    x: The optimal solution.
    s: The optimal slack variables.
    z: The optimal dual variables.
  """

  x = x0
  s = s0
  z = z0

  for _ in range(max_iter):
    # 1. Evaluate function, gradient, and Hessian at current point
    H = hessian_f(x)
    grad = grad_f(x)

    # 2. Compute residuals
    r_d = grad + A.T @ z + G.T @ s  # Dual residual
    r_p = A @ x - b                  # Primal equality residual
    r_g = s + G @ x - h              # Primal inequality residual

    # 3. Check for convergence
    if np.linalg.norm(r_d) <= tol and \
       np.linalg.norm(r_p) <= tol and \
       np.linalg.norm(r_g) <= tol:
      break

    # 4. Compute Nesterov-Todd scaling matrix
    W_squared = np.diag(s / z)
    W = np.sqrt(W_squared)

    # 5. Solve the KKT system for affine scaling direction
    dx_aff, dz_aff, ds_aff = solve_kkt_system(H, A, G, W, r_d, r_p, r_g)

    # 6. Calculate step size for affine scaling direction
    alpha_aff_p = min(1, np.min(-s[ds_aff < 0] / ds_aff[ds_aff < 0])) if any(ds_aff < 0) else 1
    alpha_aff_d = min(1, np.min(-z[dz_aff < 0] / dz_aff[dz_aff < 0])) if any(dz_aff < 0) else 1

    # 7. Calculate centering parameter (sigma) using Mehrotra's correction
    mu = s @ z / len(s)
    mu_aff = (s + alpha_aff_p * ds_aff) @ (z + alpha_aff_d * dz_aff) / len(s)
    sigma = (mu_aff / mu) ** 3

    # 8. Solve the KKT system for the final search direction
    r_g_corr = r_g + (1 / alpha_aff_p) * ds_aff * dz_aff - sigma * mu * np.ones_like(s)
    dx, dz, ds = solve_kkt_system(H, A, G, W, r_d, r_p, r_g_corr)

    # 9. Calculate step size for the final search direction
    alpha_p = min(1, 0.99 * np.min(-s[ds < 0] / ds[ds < 0])) if any(ds < 0) else 1
    alpha_d = min(1, 0.99 * np.min(-z[dz < 0] / dz[dz < 0])) if any(dz < 0) else 1

    # 10. Update variables
    x += alpha_p * dx
    s += alpha_p * ds
    z += alpha_d * dz

  return x, s, z