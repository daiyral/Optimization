import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Version 2. with cvxpy optimization for reducing the problem of finding closest point of an ellipsoid to the origin to constrained linear regression (function called reduce_to_constrained_linear_regression)
# Danny Aibinder: 318239639
# Bradley Feitsvaig: 311183073

def is_nonzero(v):
    return np.linalg.norm(v) > 0

def is_correct_dim(v, d):
    return v.shape == (d,)

def is_in_sp(v1, v2):
    # if they are in the same plane, the cross product will be 0
    return np.linalg.norm(np.cross(v1, v2)) == 0

def is_in_sp_pq(v, p, q):
    # if v is outside the span, the rank will increase else it will not so we compare ranks with and without v
    return np.linalg.matrix_rank([p, q]) == np.linalg.matrix_rank([p, q, v])


def rotate(w, u):
    """
    Calculates a rotation matrix that rotates w normalized to u normalized
    Args:
        w: R^2\\{0}
        u: R^2\\{0}
    Returns:
        U: rotation matrix in SO(2)
    """
    assert is_nonzero(w), "w must not be zero"
    assert is_nonzero(u), "u must not be zero"
    
    # normalize w and u
    w = w / np.linalg.norm(w)
    u = u / np.linalg.norm(u)

    Ju = np.array([-u[1], u[0]]) # Ju = (-u2, u1)
    Jw = np.array([-w[1], w[0]]) # Jw = (-w2, w1)

    # [u Ju] [w Jw]^T
    U = np.column_stack((u, Ju)) @ np.column_stack((w, Jw)).T
    return U


def V_from_V(p, q, v):
    """
    Calculates a matrix that can move a point v after rotation, so it stays in the plane made by p and q
    Args:
        p: R^d\\{0}
        q: R^d\\sp{p}
        v: in sp{p,q}
    Returns:
        V(v): (d,2) matrix
    """
    assert is_nonzero(p), "p must not be zero"
    assert not is_in_sp(q,p), "q must not be in sp(p)"
    assert is_in_sp_pq(v,p,q), "v must be in sp(p,q)"

    # calculate F
    p_norm = p / np.linalg.norm(p)
    pq = (np.linalg.norm(p) ** 2) * q - (q.T @ p) * p
    pq_norm = pq / np.linalg.norm(pq)
    F = np.column_stack((p_norm, pq_norm))  # (d,2)

    # calculate p', q', v'
    pts = np.column_stack((p, q, v))  # (d,3)
    pts_prime = F.T @ pts  # (2,3)
    p_prime, q_prime, v_prime = pts_prime[:, 0], pts_prime[:, 1], pts_prime[:, 2]

    # calculate U
    if not np.array_equal(v_prime, q_prime): # v' != q'
        U = rotate(p_prime - q_prime, v_prime - q_prime)
    else:
        U = np.zeros((2, 2))

    # calculate c
    denom = np.linalg.norm(p_prime - q_prime)
    c = np.linalg.norm(v_prime - q_prime) / denom

    # calculate V(v)
    I = np.eye(2)
    M = np.column_stack((c * U @ p_prime, (c * U - I) @ q_prime)) # [cU p', (cU - I) q']
    prime_mat = np.column_stack((p_prime, q_prime)) # [p', q']
    prime_mat_inv = np.linalg.inv(prime_mat) # [p', q']^-1
    V_v = np.linalg.norm(p - q) * F @ M @ prime_mat_inv # ||p - q||F[cU p', (cU - I) q'][p', q']^-1

    return V_v


def PQBU(p, q, z):
    """
    describe all possible locations that z can move when moving triangle defined by p,q,z (locus of z) where p,q are fixed on the original line
    Args:
        z: R^3
        p: R^3\\{0}
        q: R^3\\sp{p}
    Returns:
        5 tuple (P, Q, B, U, u):
            P, Q, B, U: 3 x 2 matrix
            u: R^3 vector
    """
    assert is_correct_dim(p, 3), "p must be in R^3"
    assert is_correct_dim(q, 3), "q must be in R^3"
    assert is_correct_dim(z, 3), "z must be in R^3"
    assert is_nonzero(p), "p must not be zero"
    assert not is_in_sp(q,p), "q must not be in sp(p)"

    P = V_from_V(p, q, p)
    Q = V_from_V(p, q, q)

    # calculate b the proj of z onto the line q + span{p - q}
    pq = p - q
    norm_pq = np.linalg.norm(pq)
    proj_mat = np.outer(pq, pq) / (norm_pq ** 2)
    b = q + proj_mat @ (z - q)

    B = V_from_V(p, q, b)

    # v orthogonal to sp{p, q}
    v = np.cross(p, q)

    scale = - (np.linalg.norm(z - b) / norm_pq) # -||z-b|| / ||p-q||
    # [v]_x
    v_cross = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])
    U = scale * (v_cross @ (P - Q)) # U= -(||z - b|| / ||p - q||) [v]_x (P - Q)

    # u = ||z-b|| * v
    u = np.linalg.norm(z - b) * v

    return P, Q, B, U, u


def FG(p, q, z, l):
    """
    Calculate how z moves relative to l under all rots/trans of triangle (p, q, z)
    Args:
        p: R^3\\{0}
        l: R^3\\{0}
        q: R^3\\sp{p}
        z: R^3
    Returns:
        3 tuple (F, G, c):
            F, G: 2x2 matrix
            c: c >= 0 scalar
    """
    assert is_correct_dim(p,3), "p must be in R^3"
    assert is_correct_dim(q,3), "q must be in R^3"
    assert is_correct_dim(z,3), "z must be in R^3"
    assert is_correct_dim(l,3), "l must be in R^3"
    assert is_nonzero(p), "p must not be zero"
    assert is_nonzero(l), "l must not be zero"
    assert not is_in_sp(q,p), "q must not be in sp(p)"

    # norm l
    norm_l = np.linalg.norm(l)
    l = l / norm_l
    l1, l2, l3 = l
    if abs(l1) < 1e-14 and abs(l2) < 1e-14: # (l1, l2) = (0,0)
        l_orth = np.array([0.0, 1.0, 0.0]) # (0,1,0)^T
    else:
        denom = np.sqrt(l1 ** 2 + l2 ** 2) # ||(l1, l2)||
        l_orth = np.array([-l2, l1, 0.0]) / denom # (-l2, l1, 0)^T /||(l2,l1)||

    l_cross = np.cross(l, l_orth) # l x l_orth
    L = np.column_stack((l_orth, l_cross)) # [l_orth, l x l_orth] (3x2)

    P, Q, B, U, u = PQBU(p, q, z)

    # c = ||L^Tu||
    c_vec = L.T @ u
    c = np.linalg.norm(c_vec)

    # check if u is zero vector
    if np.allclose(u, 0, atol=1e-14):
        H = np.eye(2)  # H = I (2x2)
    else:
        w = L.T @ u # L^T u
        H = rotate(w, np.array([0.0, 1.0]))

    # F = H L^T B
    F = H @ (L.T @ B)
    # G = H L^T U
    G = H @ (L.T @ U)

    return F, G, c


# Solves Fx - Gy = 0 under ||x|| <= 1 This retruns the x point on the ellipsoid that is closest to the origin
def reduce_to_constrained_linear_regression(F, G, y=np.array([1.0, 0.0])):
    b = G @ y 
    x = cp.Variable(2) # (x1, x2) as a variable to solve
    objective = cp.Minimize(cp.norm2(F @ x - b)) # minimize ||Fx - b|| L2 norm
    constraints = [cp.norm(x, 2) <= 1] # ||x|| <= 1
    problem = cp.Problem(objective, constraints) 
    problem.solve()

    return x.value, problem.value


if __name__ == "__main__":
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    z = np.array([0.5, 0.5, 1.0])
    l = np.array([1.0, 1.0, 1.0])
    F, G, _ = FG(p, q, z, l) # get FG
    x_opt, val = reduce_to_constrained_linear_regression(F, G) # reduce to linear regression

    print("Optimal x:", x_opt)
    print("Objective value:", val)
    

    # Why does solving this FG problem find the closest point on an ellipsoid to the origin?
        # we represent the elipsoid as ||Fx - Gy|| where F rotates and scales the elipsoid and Gy shifts the elipsoid. This is similar to Ax - b.
        # we can define a circle as Fx, if F is a orthogonal matrix, otherwise it is a elipsoid
        # the points on the ellipsoid are represented by Fx, the optimization finds the point Fx on a elipsoid shifted by Gy to the origin under the ||x|| <= 1 constraint to ensure points within the ellipsoid.
        # so minimizing ||Fx - Gy|| finds the x point on an shifted ellipsoid closest to the origin with the objective of minimizing euclidean distance of a poitn to the origin. this is similar to the constrained linear regression.

    # Why does solving this FG problem solve a 2D PnP problem?
        # FG captures motions of 3d point while preserving geometry of a triangle. 
        # we project the 3D points into 2D space using the F and G matrices, which are the rotation and translation of the triangle in 3D space.
        # we solve || Fx - Gy|| where Fx is projected points from the moving triangle and Gy is fixed points.
        # this is similar to 3d points points projection(Fx) and line l from 2d image (Gy).
        # So we reduce the 3D geometric constraint to 2d convex by optimization projection over unit variable x. 
        # solving this minimizes how far the project point Fx is from the observed line Gy, the PnP problem of finding camera pose.
