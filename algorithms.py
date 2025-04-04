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
    Ju = np.array([-u[1], u[0]])  # Ju = (-u2, u1)
    Jw = np.array([-w[1], w[0]])  # Jw = (-w2, w1)

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


def PQBU(p, q, z):
    """
    Compute the tuple (P, Q, B, U, u) according to Algorithm 3.

    Args:
        p, q, z: 3-dimensional vectors (np.ndarray of shape (3,)).
                 p must be nonzero and q must not be collinear with p.

    Returns:
        A tuple (P, Q, B, U, u) where:
            P, Q, B, U: 3 x 2 matrices.
            u: 3-dimensional vector.
    """
    # check inputs
    assert p.shape == (3,), "p must be a 3-dimensional vector."
    assert q.shape == (3,), "q must be a 3-dimensional vector."
    assert z.shape == (3,), "z must be a 3-dimensional vector."
    assert np.linalg.norm(p) > 1e-14, "p must be nonzero."
    # Ensure p and q are not collinear by checking the cross product.
    assert np.linalg.norm(np.cross(p, q)) > 1e-14, "q must not be collinear with p."

    # step 1: Compute P = V_from_V(p, q, p)
    P = V_from_V(p, q, p)

    # step 2: Compute Q = V_from_V(p, q, q)
    Q = V_from_V(p, q, q)

    # step 3: Compute b, the projection of z onto the line q + span{p - q}
    pq = p - q
    norm_pq = np.linalg.norm(pq)
    proj_mat = np.outer(pq, pq) / (norm_pq ** 2)
    b = q + proj_mat @ (z - q)

    # step 4: Compute B = V_from_V(p, q, b)
    B = V_from_V(p, q, b)

    # Step 5: v = p x q (cross product, orthogonal to plane sp{p, q})
    v = np.cross(p, q)

    # Step 6: U := - (||z - b|| / ||p - q||) [v]_x (P - Q)
    scale = - (np.linalg.norm(z - b) / norm_pq)
    # Build the 3x3 cross-product matrix [v]_x
    v_cross = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])
    U = scale * (v_cross @ (P - Q))

    # Step 7: u = ||z-b|| * v
    u = np.linalg.norm(z - b) * v

    return P, Q, B, U, u


def FG(p, q, z, l):
    """
    Compute the tuple (F, G, c) according to Algorithm 4.

    Args:
        p, q, z: 3-dimensional vectors (np.ndarray of shape (3,)).
                 p must be nonzero, and q must not be collinear with p.
        l      : 3-dimensional vector (np.ndarray of shape (3,)) used to define
                 an orthonormal basis for projection.

    Returns:
        A tuple (F, G, c) where:
            F, G: 2 x 2 matrices
            c   : a nonnegative scalar (default 0 here, or compute per your reference).
    """
    # check inputs
    assert p.shape == (3,), "p must be a 3-dimensional vector."
    assert q.shape == (3,), "q must be a 3-dimensional vector."
    assert z.shape == (3,), "z must be a 3-dimensional vector."
    assert l.shape == (3,), "l must be a 3-dimensional vector."
    assert np.linalg.norm(p) > 1e-14, "p must be nonzero."
    assert np.linalg.norm(l) > 1e-14, "l must be nonzero."
    # Ensure p and q are not collinear by checking the cross product.
    assert np.linalg.norm(np.cross(p, q)) > 1e-14, "q must not be collinear with p."

    # Step 1: l := l / ||l||
    norm_l = np.linalg.norm(l)
    l = l / norm_l

    # Step 2-5: if (l1, l2) = (0,0) then l_perp = (0,1,0)^T
    #         else l_perp = (-l2, l1, 0)^T /||(l2,l1)||
    l1, l2, l3 = l
    if abs(l1) < 1e-14 and abs(l2) < 1e-14:
        # If (l1, l2) = (0,0), then set l_orth = (0,1,0)^T.
        l_orth = np.array([0.0, 1.0, 0.0])
    else:
        # Otherwise, l_orth = (-l2, l1, 0)^T /||(l2,l1)||.
        denom = np.sqrt(l1 ** 2 + l2 ** 2)
        l_orth = np.array([-l2, l1, 0.0]) / denom

    # Step 6: Form the 3x2 matrix L with left column = l_orth and right column = l_cross.
    l_cross = np.cross(l, l_orth)
    L = np.column_stack((l_orth, l_cross))

    # Step 7: (P, Q, B, U, u) = PQBU(p, q, z)
    P, Q, B, U, u = PQBU(p, q, z)

    # Step 8: c:=||L^Tu||
    c_vec = L.T @ u
    c = np.linalg.norm(c_vec)

    # Step 9: Check if u is the zero vector.
    if np.allclose(u, 0, atol=1e-14):
        H = np.eye(2)  # Step 10: If u is zero, set H to the 2x2 identity matrix.
    else:
        # step 11-12: else H := ROTATE(L^T u, (0,1)^T)
        w = L.T @ u
        H = rotate(w, np.array([0.0, 1.0]))

    # Step 13: F := H L^T B
    F = H @ (L.T @ B)

    # Step 14: G := H L^T U
    G = H @ (L.T @ U)

    return F, G, c
