import numpy as np


def rotate(w, u):
    """
    Calculates a rotation matrix that rotates w normalized to Uw/||w|| = u/||u||.
    Args:
        w: vector in R^2\{0}.
        u: vector in R^2\{0}.

    Returns:
        U: rotation matrix in SO(2).
    """
    assert np.linalg.norm(w) > 0, "norm of w must not be zero"
    
    # step 1, normalize w and u
    w = w / np.linalg.norm(w)
    u = u / np.linalg.norm(u)

    # step 2 j matrix
    Ju = np.array([-u[1], u[0]]) # Ju = (-u2, u1)
    Jw = np.array([-w[1], w[0]]) # Jw = (-w2, w1)

    # step 3 [u Ju] [w Jw]^T
    U = np.column_stack((u, Ju)) @ np.column_stack((w, Jw)).T
    return U

def V_from_V(p, q, v):
    """
    Compute V(v) according to Algorithm 2.

    Args:
    p, q, v: d-dimensional vectors

    Returns:
    V(v): A d x 2 matrix
    """
    assert p.shape == q.shape == v.shape, "p, q, v must be vectors of same dimension"
    assert np.linalg.norm(p) > 0, "p must be non-zero"
    
    # step 1 calculate F
    p_norm = p / np.linalg.norm(p)
    pq = (np.linalg.norm(p) ** 2) * q - (q.T @ p) * p
    pq_norm = pq / np.linalg.norm(pq)
    F = np.column_stack((p_norm, pq_norm))  # (d,2)
    
    # step 3 calculate transformed coordinates
    pts = np.column_stack((p, q, v))  # (d,3)
    pts_prime = F.T @ pts  # (2,3)
    p_prime, q_prime, v_prime = pts_prime[:, 0], pts_prime[:, 1], pts_prime[:, 2]
    
    # step 3-6 calculate U
    if not np.allclose(v_prime, q_prime):
        U = rotate(p_prime - q_prime, v_prime - q_prime)
    else:
        U = np.zeros((2, 2))
    
    # step 7 calculate c
    denom = np.linalg.norm(p_prime - q_prime)
    if denom != 0:
        c = np.linalg.norm(v_prime - q_prime) / denom 
    else:
        c = 0

    
    # step 8 calculate V(v)
    I = np.eye(2)
    M = np.column_stack((c * U @ p_prime, (c * U - I) @ q_prime))
    prime_mat = np.column_stack((p_prime, q_prime))
    prime_mat_inv = np.linalg.inv(prime_mat)
    V_v = np.linalg.norm(p - q) * F @ M @ prime_mat_inv
    
    return V_v

