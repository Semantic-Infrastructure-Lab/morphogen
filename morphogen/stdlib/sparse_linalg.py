"""Sparse Linear Algebra domain implementation.

This module provides sparse matrix operations and iterative solvers for large-scale
linear systems. Essential for PDEs, circuit simulation, graph algorithms, and optimization.

Supported Operations:
- Sparse matrix formats (CSR, CSC, COO)
- Iterative solvers (CG, BiCGSTAB, GMRES)
- Sparse factorizations (incomplete Cholesky, incomplete LU)
- Discrete operators (Laplacian, gradient, divergence)
"""

from typing import Optional, Tuple, Callable, Union
import numpy as np

from morphogen.core.operator import operator, OpCategory
from scipy import sparse
from scipy.sparse import linalg as sp_linalg


# ============================================================================
# SPARSE MATRIX CREATION
# ============================================================================

@operator(
    domain="sparse_linalg",
    category=OpCategory.CONSTRUCT,
    signature="(data: Union[ndarray, Tuple], shape: Optional[Tuple[int, int]]) -> csr_matrix",
    deterministic=True,
    doc="Create CSR (Compressed Sparse Row) matrix"
)
def csr_matrix(
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    shape: Optional[Tuple[int, int]] = None
) -> sparse.csr_matrix:
    """Create CSR (Compressed Sparse Row) matrix.

    CSR format is efficient for:
    - Row slicing
    - Matrix-vector products
    - Arithmetic operations

    Args:
        data: Either dense array or tuple (data, (row_ind, col_ind))
        shape: Shape of matrix (required if data is tuple)

    Returns:
        CSR sparse matrix

    Example:
        # From dense array
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        A_csr = sparse_linalg.csr_matrix(A_dense)

        # From coordinates
        data = np.array([1, 2, 3, 4, 5])
        rows = np.array([0, 0, 1, 2, 2])
        cols = np.array([0, 2, 1, 0, 2])
        A_csr = sparse_linalg.csr_matrix((data, (rows, cols)), shape=(3, 3))
    """
    if isinstance(data, tuple):
        if shape is None:
            raise ValueError("Shape must be specified when using (data, (row, col)) format")
        return sparse.csr_matrix(data, shape=shape)
    else:
        return sparse.csr_matrix(data)


@operator(
    domain="sparse_linalg",
    category=OpCategory.CONSTRUCT,
    signature="(data: Union[ndarray, Tuple], shape: Optional[Tuple[int, int]]) -> csc_matrix",
    deterministic=True,
    doc="Create CSC (Compressed Sparse Column) matrix"
)
def csc_matrix(
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    shape: Optional[Tuple[int, int]] = None
) -> sparse.csc_matrix:
    """Create CSC (Compressed Sparse Column) matrix.

    CSC format is efficient for:
    - Column slicing
    - Sparse factorizations (Cholesky, LU)
    - Column-oriented operations

    Args:
        data: Either dense array or tuple (data, (row_ind, col_ind))
        shape: Shape of matrix (required if data is tuple)

    Returns:
        CSC sparse matrix
    """
    if isinstance(data, tuple):
        if shape is None:
            raise ValueError("Shape must be specified when using (data, (row, col)) format")
        return sparse.csc_matrix(data, shape=shape)
    else:
        return sparse.csc_matrix(data)


@operator(
    domain="sparse_linalg",
    category=OpCategory.CONSTRUCT,
    signature="(data: Union[ndarray, Tuple], shape: Optional[Tuple[int, int]]) -> coo_matrix",
    deterministic=True,
    doc="Create COO (Coordinate) matrix"
)
def coo_matrix(
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    shape: Optional[Tuple[int, int]] = None
) -> sparse.coo_matrix:
    """Create COO (Coordinate) matrix.

    COO format is efficient for:
    - Incremental construction
    - Converting to other formats

    Args:
        data: Either dense array or tuple (data, (row_ind, col_ind))
        shape: Shape of matrix (required if data is tuple)

    Returns:
        COO sparse matrix
    """
    if isinstance(data, tuple):
        if shape is None:
            raise ValueError("Shape must be specified when using (data, (row, col)) format")
        return sparse.coo_matrix(data, shape=shape)
    else:
        return sparse.coo_matrix(data)


# ============================================================================
# ITERATIVE SOLVERS
# ============================================================================

@operator(
    domain="sparse_linalg",
    category=OpCategory.TRANSFORM,
    signature="(A: spmatrix, b: ndarray, x0: Optional[ndarray], tol: float, maxiter: Optional[int]) -> Tuple[ndarray, int, float]",
    deterministic=True,
    doc="Solve Ax=b using Conjugate Gradient method"
)
def solve_cg(
    A: sparse.spmatrix,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    maxiter: Optional[int] = None
) -> Tuple[np.ndarray, int, float]:
    """Solve Ax=b using Conjugate Gradient method.

    CG is optimal for symmetric positive-definite matrices.
    Convergence is guaranteed for SPD systems.

    Args:
        A: Sparse matrix (must be symmetric positive-definite)
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        tol: Convergence tolerance
        maxiter: Maximum iterations (default: None = len(b))

    Returns:
        Tuple of (solution, iterations, residual_norm)

    Example:
        # Solve Poisson equation on 2D grid
        A = laplacian_2d(64, 64)
        b = np.random.rand(64 * 64)
        x, iters, resid = sparse_linalg.solve_cg(A, b)
    """
    if x0 is None:
        x0 = np.zeros_like(b)

    # Track iterations with callback
    iteration_count = [0]

    def callback(xk):
        iteration_count[0] += 1

    # Conjugate Gradient solver (using rtol not tol)
    result, info = sp_linalg.cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, callback=callback, atol=0)
    iterations = iteration_count[0]

    if info > 0:
        import warnings
        warnings.warn(f"CG did not converge after {info} iterations")
    elif info < 0:
        raise ValueError(f"CG illegal input or breakdown (info={info})")

    # Compute final residual
    residual = np.linalg.norm(b - A @ result)

    return result, iterations, residual


@operator(
    domain="sparse_linalg",
    category=OpCategory.TRANSFORM,
    signature="(A: spmatrix, b: ndarray, x0: Optional[ndarray], tol: float, maxiter: Optional[int]) -> Tuple[ndarray, int, float]",
    deterministic=True,
    doc="Solve Ax=b using BiCGSTAB (Biconjugate Gradient Stabilized)"
)
def solve_bicgstab(
    A: sparse.spmatrix,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    maxiter: Optional[int] = None
) -> Tuple[np.ndarray, int, float]:
    """Solve Ax=b using BiCGSTAB (Biconjugate Gradient Stabilized).

    BiCGSTAB is for general nonsymmetric matrices.
    More stable than BiCG but may still fail for some matrices.

    Args:
        A: Sparse matrix (can be nonsymmetric)
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        tol: Convergence tolerance
        maxiter: Maximum iterations (default: None = len(b))

    Returns:
        Tuple of (solution, iterations, residual_norm)

    Example:
        # Solve nonsymmetric circuit equation
        A = circuit_admittance_matrix(n_nodes=1000)
        b = current_sources
        x, iters, resid = sparse_linalg.solve_bicgstab(A, b)
    """
    if x0 is None:
        x0 = np.zeros_like(b)

    # Track iterations with callback
    iteration_count = [0]

    def callback(xk):
        iteration_count[0] += 1

    result, info = sp_linalg.bicgstab(A, b, x0=x0, rtol=tol, maxiter=maxiter, callback=callback, atol=0)
    iterations = iteration_count[0]

    if info > 0:
        import warnings
        warnings.warn(f"BiCGSTAB did not converge after {info} iterations")
    elif info < 0:
        raise ValueError(f"BiCGSTAB illegal input or breakdown (info={info})")

    residual = np.linalg.norm(b - A @ result)

    return result, iterations, residual


@operator(
    domain="sparse_linalg",
    category=OpCategory.TRANSFORM,
    signature="(A: spmatrix, b: ndarray, x0: Optional[ndarray], tol: float, restart: int, maxiter: Optional[int]) -> Tuple[ndarray, int, float]",
    deterministic=True,
    doc="Solve Ax=b using GMRES (Generalized Minimal Residual)"
)
def solve_gmres(
    A: sparse.spmatrix,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    restart: int = 20,
    maxiter: Optional[int] = None
) -> Tuple[np.ndarray, int, float]:
    """Solve Ax=b using GMRES (Generalized Minimal Residual).

    GMRES is for general nonsymmetric matrices.
    Very robust but memory-intensive (uses restart parameter).

    Args:
        A: Sparse matrix
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        tol: Convergence tolerance
        restart: Number of iterations before restart (default 20)
        maxiter: Maximum iterations (default: None = len(b))

    Returns:
        Tuple of (solution, iterations, residual_norm)

    Example:
        # Solve fluid dynamics equation (Navier-Stokes)
        A = fluid_jacobian_matrix()
        b = momentum_rhs
        x, iters, resid = sparse_linalg.solve_gmres(A, b, restart=30)
    """
    if x0 is None:
        x0 = np.zeros_like(b)

    # Track iterations
    iteration_count = [0]

    def callback(rk):
        iteration_count[0] += 1

    result, info = sp_linalg.gmres(
        A, b, x0=x0, rtol=tol, restart=restart, maxiter=maxiter, callback=callback, atol=0
    )
    iterations = iteration_count[0]

    if info > 0:
        import warnings
        warnings.warn(f"GMRES did not converge after {info} iterations")
    elif info < 0:
        raise ValueError(f"GMRES illegal input or breakdown (info={info})")

    residual = np.linalg.norm(b - A @ result)

    return result, iterations, residual


@operator(
    domain="sparse_linalg",
    category=OpCategory.TRANSFORM,
    signature="(A: spmatrix, b: ndarray, method: str, **kwargs) -> ndarray",
    deterministic=True,
    doc="Generic sparse solver with automatic method selection"
)
def solve_sparse(
    A: sparse.spmatrix,
    b: np.ndarray,
    method: str = "auto",
    **kwargs
) -> tuple:
    """Generic sparse solver with automatic method selection.

    Args:
        A: Sparse matrix
        b: Right-hand side vector
        method: Solver method ("auto", "cg", "bicgstab", "gmres", "direct")
        **kwargs: Additional arguments passed to solver

    Returns:
        Tuple of (solution_vector, info_dict) where info_dict contains solver metadata.

    Example:
        x, info = sparse_linalg.solve_sparse(A, b, method="auto")
    """
    if method == "auto":
        # Auto-select based on matrix properties
        if A.shape[0] < 1000:
            # Small system → use direct solver
            method = "direct"
        elif np.allclose(A.todense(), A.todense().T):
            # Symmetric → use CG
            method = "cg"
        else:
            # Nonsymmetric → use BiCGSTAB
            method = "bicgstab"

    if method == "cg":
        x, iters, residual = solve_cg(A, b, **kwargs)
        info = {"method": "cg", "converged": residual < 1e-10, "iterations": iters, "residual": residual}
    elif method == "bicgstab":
        x, iters, residual = solve_bicgstab(A, b, **kwargs)
        info = {"method": "bicgstab", "converged": residual < 1e-10, "iterations": iters, "residual": residual}
    elif method == "gmres":
        x, iters, residual = solve_gmres(A, b, **kwargs)
        info = {"method": "gmres", "converged": residual < 1e-10, "iterations": iters, "residual": residual}
    elif method == "direct":
        x = sp_linalg.spsolve(A, b)
        info = {"method": "direct", "converged": True, "iterations": 1, "residual": 0.0}
    else:
        raise ValueError(f"Unknown solver method: {method}")

    return x, info


# ============================================================================
# SPARSE FACTORIZATIONS
# ============================================================================

@operator(
    domain="sparse_linalg",
    category=OpCategory.TRANSFORM,
    signature="(A: spmatrix) -> LinearOperator",
    deterministic=True,
    doc="Compute incomplete Cholesky factorization (preconditioner)"
)
def incomplete_cholesky(A: sparse.spmatrix) -> sparse.linalg.LinearOperator:
    """Compute incomplete Cholesky factorization (preconditioner).

    Useful as preconditioner for CG solver.

    Args:
        A: Sparse symmetric positive-definite matrix

    Returns:
        Linear operator representing incomplete Cholesky factor

    Example:
        A = laplacian_2d(128, 128)
        M = sparse_linalg.incomplete_cholesky(A)
        x, info = sp_linalg.cg(A, b, M=M)  # Preconditioned CG
    """
    try:
        # Attempt incomplete Cholesky
        ilu = sp_linalg.spilu(A.tocsc())
        return sp_linalg.LinearOperator(A.shape, ilu.solve)
    except (RuntimeError, ValueError):
        # Factorization failed (e.g. singular / not factorizable) -> fall back to incomplete LU
        import warnings
        warnings.warn("Incomplete Cholesky failed, using incomplete LU instead")
        return incomplete_lu(A)


@operator(
    domain="sparse_linalg",
    category=OpCategory.TRANSFORM,
    signature="(A: spmatrix) -> LinearOperator",
    deterministic=True,
    doc="Compute incomplete LU factorization (preconditioner)"
)
def incomplete_lu(A: sparse.spmatrix) -> sparse.linalg.LinearOperator:
    """Compute incomplete LU factorization (preconditioner).

    Useful as preconditioner for BiCGSTAB/GMRES solvers.

    Args:
        A: Sparse matrix

    Returns:
        Linear operator representing incomplete LU factors

    Example:
        A = circuit_matrix(1000)
        M = sparse_linalg.incomplete_lu(A)
        x, info = sp_linalg.bicgstab(A, b, M=M)  # Preconditioned BiCGSTAB
    """
    ilu = sp_linalg.spilu(A.tocsc())
    return sp_linalg.LinearOperator(A.shape, ilu.solve)


# ============================================================================
# DISCRETE OPERATORS
# ============================================================================

@operator(
    domain="sparse_linalg",
    category=OpCategory.CONSTRUCT,
    signature="(n: int, bc: str) -> csr_matrix",
    deterministic=True,
    doc="Create 1D Laplacian operator (discrete second derivative)"
)
def laplacian_1d(n: int, bc: str = "dirichlet") -> sparse.csr_matrix:
    """Create 1D Laplacian operator (discrete second derivative).

    Discretizes: d²u/dx² using finite differences.

    Args:
        n: Number of grid points
        bc: Boundary conditions ("dirichlet", "neumann", "periodic")

    Returns:
        Sparse CSR matrix of shape (n, n)

    Example:
        # Solve heat equation: du/dt = alpha * d²u/dx²
        L = sparse_linalg.laplacian_1d(100)
        u = np.sin(np.linspace(0, np.pi, 100))
        dudt = alpha * L @ u
    """
    if bc == "dirichlet":
        # Standard centered difference: [1, -2, 1]
        diags = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
        L = sparse.diags(diags, [-1, 0, 1], shape=(n, n), format='csr')

    elif bc == "neumann":
        # Neumann BC: du/dx = 0 at boundaries
        diags = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
        L = sparse.diags(diags, [-1, 0, 1], shape=(n, n), format='csr')
        # Modify boundary rows
        L[0, 0] = -1
        L[-1, -1] = -1

    elif bc == "periodic":
        # Periodic BC: u(0) = u(n)
        diags = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
        L = sparse.diags(diags, [-1, 0, 1], shape=(n, n), format='lil')
        L[0, -1] = 1
        L[-1, 0] = 1
        L = L.tocsr()

    else:
        raise ValueError(f"Unknown boundary condition: {bc}")

    return L


@operator(
    domain="sparse_linalg",
    category=OpCategory.CONSTRUCT,
    signature="(nx: int, ny: int, bc: str) -> csr_matrix",
    deterministic=True,
    doc="Create 2D Laplacian operator (discrete Laplacian)"
)
def laplacian_2d(nx: int, ny: int, bc: str = "dirichlet") -> sparse.csr_matrix:
    """Create 2D Laplacian operator (discrete Laplacian).

    Discretizes: ∇²u = ∂²u/∂x² + ∂²u/∂y² using finite differences.

    Args:
        nx: Number of grid points in x-direction
        ny: Number of grid points in y-direction
        bc: Boundary conditions ("dirichlet", "neumann", "periodic")

    Returns:
        Sparse CSR matrix of shape (nx*ny, nx*ny)

    Example:
        # Solve Poisson equation: ∇²u = f
        L = sparse_linalg.laplacian_2d(64, 64)
        f = np.random.rand(64 * 64)  # Source term
        u = sparse_linalg.solve_cg(L, -f)[0]
        u_2d = u.reshape(64, 64)
    """
    n = nx * ny

    if bc == "dirichlet":
        # 5-point stencil: [-1, -1, 4, -1, -1]
        main_diag = 4 * np.ones(n)
        x_diag = -np.ones(n - 1)
        y_diag = -np.ones(n - nx)

        # Handle x-boundary (don't connect across x-boundary)
        x_diag[nx-1::nx] = 0

        diags = [y_diag, x_diag, main_diag, x_diag, y_diag]
        L = sparse.diags(diags, [-nx, -1, 0, 1, nx], shape=(n, n), format='csr')

    elif bc == "neumann":
        # Similar to Dirichlet but with modified boundary handling
        main_diag = 4 * np.ones(n)
        x_diag = -np.ones(n - 1)
        y_diag = -np.ones(n - nx)

        x_diag[nx-1::nx] = 0

        diags = [y_diag, x_diag, main_diag, x_diag, y_diag]
        L = sparse.diags(diags, [-nx, -1, 0, 1, nx], shape=(n, n), format='csr')

        # Modify boundary points (simplified)
        for i in range(nx):
            L[i, i] = 3  # Top boundary
            L[n-1-i, n-1-i] = 3  # Bottom boundary
        for i in range(0, n, nx):
            L[i, i] = 3  # Left boundary
            L[i+nx-1, i+nx-1] = 3  # Right boundary

    elif bc == "periodic":
        # Periodic in both directions
        main_diag = 4 * np.ones(n)
        x_diag = -np.ones(n - 1)
        y_diag = -np.ones(n - nx)

        diags = [y_diag, x_diag, main_diag, x_diag, y_diag]
        L = sparse.diags(diags, [-nx, -1, 0, 1, nx], shape=(n, n), format='lil')

        # Add periodic wrapping
        for i in range(ny):
            L[i*nx, (i+1)*nx-1] = -1  # x-periodic
            L[(i+1)*nx-1, i*nx] = -1

        for i in range(nx):
            L[i, n-nx+i] = -1  # y-periodic
            L[n-nx+i, i] = -1

        L = L.tocsr()

    else:
        raise ValueError(f"Unknown boundary condition: {bc}")

    return L


def build_laplacian_2d(nx: int, ny: int, dx: float = 1.0, bc: str = "dirichlet") -> sparse.csr_matrix:
    """Alias for laplacian_2d with optional grid spacing dx.

    Args:
        nx: Number of grid points in x-direction
        ny: Number of grid points in y-direction
        dx: Grid spacing (divides the Laplacian by dx²)
        bc: Boundary conditions ("dirichlet", "neumann", "periodic")

    Returns:
        Sparse CSR matrix of shape (nx*ny, nx*ny)
    """
    L = laplacian_2d(nx, ny, bc=bc)
    if dx != 1.0:
        L = L / (dx ** 2)
    return L


@operator(
    domain="sparse_linalg",
    category=OpCategory.CONSTRUCT,
    signature="(nx: int, ny: int) -> Tuple[csr_matrix, csr_matrix]",
    deterministic=True,
    doc="Create 2D gradient operators (discrete gradient)"
)
def gradient_2d(nx: int, ny: int) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Create 2D gradient operators (discrete gradient).

    Discretizes: ∇u = (∂u/∂x, ∂u/∂y) using forward differences.

    Args:
        nx: Number of grid points in x-direction
        ny: Number of grid points in y-direction

    Returns:
        Tuple of (Gx, Gy) sparse matrices

    Example:
        # Compute gradient of scalar field
        Gx, Gy = sparse_linalg.gradient_2d(64, 64)
        u = np.random.rand(64 * 64)
        ux = Gx @ u
        uy = Gy @ u
    """
    n = nx * ny

    # Gradient in x-direction (forward difference)
    x_diag = -np.ones(n)
    x_diag_p1 = np.ones(n - 1)
    x_diag_p1[nx-1::nx] = 0  # Don't wrap across x-boundary

    Gx = sparse.diags([x_diag, x_diag_p1], [0, 1], shape=(n, n), format='csr')

    # Gradient in y-direction (forward difference)
    y_diag = -np.ones(n)
    y_diag_pnx = np.ones(n - nx)

    Gy = sparse.diags([y_diag, y_diag_pnx], [0, nx], shape=(n, n), format='csr')

    return Gx, Gy


@operator(
    domain="sparse_linalg",
    category=OpCategory.CONSTRUCT,
    signature="(nx: int, ny: int) -> csr_matrix",
    deterministic=True,
    doc="Create 2D divergence operator (discrete divergence)"
)
def divergence_2d(nx: int, ny: int) -> sparse.csr_matrix:
    """Create 2D divergence operator (discrete divergence).

    Discretizes: ∇·v = ∂vx/∂x + ∂vy/∂y using backward differences.

    Note: For staggered grids, divergence is -transpose of gradient.

    Args:
        nx: Number of grid points in x-direction
        ny: Number of grid points in y-direction

    Returns:
        Sparse CSR matrix

    Example:
        # Compute divergence of vector field
        div = sparse_linalg.divergence_2d(64, 64)
        vx = np.random.rand(64 * 64)
        vy = np.random.rand(64 * 64)
        divV = div @ np.concatenate([vx, vy])  # Assuming interleaved storage
    """
    Gx, Gy = gradient_2d(nx, ny)
    # Divergence is negative transpose of gradient
    return -Gx.T


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Sparse matrix creation
    'csr_matrix',
    'csc_matrix',
    'coo_matrix',

    # Iterative solvers
    'solve_cg',
    'solve_bicgstab',
    'solve_gmres',
    'solve_sparse',

    # Factorizations
    'incomplete_cholesky',
    'incomplete_lu',

    # Discrete operators
    'laplacian_1d',
    'laplacian_2d',
    'gradient_2d',
    'divergence_2d',
]
