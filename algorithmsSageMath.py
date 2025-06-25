# Danny Aibinder: 318239639
# Bradley Feitsvaig: 311183073

import matplotlib.pyplot as plt
import os

# Import the msolve interface functions
"""
17/02/2021
Adapted from the maple interface written by Huu Phuoc Le and Jorge Garcia-Fontan.
"""

def ToMSolve(F, finput="/tmp/in.ms"):
    """Convert a system of sage polynomials into a msolve input file.

    Inputs :
    F (list of polynomials): system of polynomial to solve
    finput (string): name of the msolve input file.

    """
    A = F[0].parent()
    assert all(A1 == A for A1 in map(parent,F)),\
            "The polynomials in the system must belong to the same polynomial ring."

    variables, char = A.variable_names(), A.characteristic()
    s = (", ".join(variables) + " \n"
            + str(char) + "\n")

    B = A.change_ring(order = 'degrevlex') 
    F2 = [ str(B(f)).replace(" ", "") for f in F ]
    if "0" in F2:
        F2.remove("0")
    s += ",\n".join(F2) + "\n"

    fd = open(finput, 'w')
    fd.write(s)
    fd.close()


def FormatOutputMSolveOnlySolutions(foutput):
    """Convert a msolve output file into solutions

    Inputs :
    foutput (string): name of the msolve output file

    Output :
        The set of solutions computed by msolve.

    """
    f = open(foutput,'r')
    s = f.read()
    s = s.replace("\n","").replace(":","")
    R = sage_eval(s)
    intervals = R[1][1]
    S   =   []
    if len(intervals) > 0:
        nvars   =   len(intervals[0])
        for sol in intervals:
            s = []
            for i in range(nvars):
                s.append((sol[i][0]+sol[i][1])/2)
            S.append(s)
    return S

def FormatOutputMSolve(foutput):
    """Convert a msolve output file into a rational parametrization 

    Inputs :
    foutput (string): name of the msolve output file

    Output :
        A rational parametrization of the zero-dimensional ideal describing
    the solutions. Note : p[i] and c[i] stand for the (i+1)-th coordinate.

    """
    f = open(foutput,'r')
    s = f.read()
    s = s.replace("\n","").replace(":","")
    R = sage_eval(s)
    A.<t> = QQ[]
    # dimension
    dim = R[0]
    if dim > 0:
        return None, None, A(-1), None, None, None, None

    # parametrization
    nvars       = R[1][1]
    qdim        = R[1][2]
    varstr      = R[1][3]
    linearform  = R[1][4]
    elim        = R[1][5][1][0]
    den         = R[1][5][1][1]
    polys       = R[1][5][1][2]
    # solutions
    intervals   = R[2][1]

    #  nvars, degquot, deg = L[1], L[2], L[5][0]
    #  varstr      =   L[3]
    #  linearform  =   L[4]

    if len(elim) > 0:
        pelim = A(elim[1])
    else:
        return None, None, A(-2), None, None, None, None

    pden, p, c = A(1), [], []
    if qdim > 0:
        pden = A(den[1])
        for l in polys:
            p.append(A(l[0][1]))
            c.append( l[1] )

    S   =   []
    if len(intervals) > 0:
        for sol in intervals:
            s = []
            for i in range(nvars):
                s.append((sol[i][0]+sol[i][1])/2)
            S.append(s)
    return [varstr, linearform, pelim, pden, p, c, S]

def GetRootsFromMSolve(foutput, param):
    """Compute rational approximation roots from an msolve output file
    The rational number is chosen in the isolating interval to be
    the smallest in bitsize.

    Inputs :
    foutput (string): name of the msolve output file

    Output :
        b (integer): error code
    Qroots : list of rationals approximations of the roots

    """

    if param == 1:
        varstr, linearform, elim, den, p, c, sols = FormatOutputMSolve(foutput)
        if elim.degree() == 0:
            return elim, [], []
        return 0, [varstr, linearform, elim, den, p, c], sols
    else:
        sols = FormatOutputMSolveOnlySolutions(foutput)
        return 0, [], sols


def MSolveRealRoots(F, fname1="/tmp/in.ms", fname2="/tmp/out.ms",
        mspath="../binary/msolve", v=0, p=1):
    """Computes the a rational approximation of the real roots
    of a system of sage polynomials using msolve. 

    Inputs :
    F (list of polynomials): system of polynomials to solve
    fname1 (string): complete name of the msolve input file used
    fname2 (string): complete name of the msolve output file used
    mspath (string): path to the msolve binary
    v (in [0,1,2]): level of msolve verbosity

    Output :
        sols (list of lists): list of rational approximation roots to the system
    represented by F.

    """

    ToMSolve(F, fname1)

    os.system(mspath +" -v " + str(v) +" -P " + str(p) +  " -f " + fname1 + " -o " + fname2)

    b, param, sols = GetRootsFromMSolve(fname2,p)

    if b == -1:
        print("System has infinitely many complex solutions")
        return []
    if b == -2:
        print("System not in generic position. You may add to your system")
        print("a random linear form of your variables and a new variable")
        return []
    #New(s) variable(s) may have been introduced at the end, for genericity purposes.	
    n = len(F[0].parent().gens())
    if p == 0:
        return [ s[:n] for s in sols ]
    else:
        return param, [ s[:n] for s in sols ]


def is_nonzero(v):
    return v.norm() > 0

def is_correct_dim(v, d):
    return len(v) == d

def is_in_sp(v1, v2):
    # if they are in the same plane, the cross product will be 0
    cross_prod = v1.cross_product(v2) if hasattr(v1, 'cross_product') else vector([0, 0, v1[0]*v2[1] - v1[1]*v2[0]])
    return cross_prod.norm() < 1e-12

def is_in_sp_pq(v, p, q):
    # if v is outside the span, the rank will increase else it will not so we compare ranks with and without v
    mat1 = matrix([p, q])
    mat2 = matrix([p, q, v])
    return mat1.rank() == mat2.rank()

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
    w = w / w.norm()
    u = u / u.norm()

    Ju = vector([-u[1], u[0]]) # Ju = (-u2, u1)
    Jw = vector([-w[1], w[0]]) # Jw = (-w2, w1)

    # [u Ju] [w Jw]^T
    U = matrix([u, Ju]).transpose() * matrix([w, Jw])
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
    assert not is_in_sp(q, p), "q must not be in sp(p)"
    assert is_in_sp_pq(v, p, q), "v must be in sp(p,q)"

    # calculate F
    p_norm = p / p.norm()
    pq = (p.norm() ** 2) * q - (q.dot_product(p)) * p
    pq_norm = pq / pq.norm()
    F = matrix([p_norm, pq_norm]).transpose()  # (d,2)

    # calculate p', q', v'
    pts = matrix([p, q, v]).transpose()  # (d,3)
    pts_prime = F.transpose() * pts  # (2,3)
    p_prime = vector(pts_prime.column(0))
    q_prime = vector(pts_prime.column(1))
    v_prime = vector(pts_prime.column(2))

    # calculate U
    if not (v_prime - q_prime).norm() < 1e-12: # v' != q'
        U = rotate(p_prime - q_prime, v_prime - q_prime)
    else:
        U = matrix(2, 2, 0)

    # calculate c
    denom = (p_prime - q_prime).norm()
    c = (v_prime - q_prime).norm() / denom

    # calculate V(v)
    I = identity_matrix(2)
    M = matrix([c * U * p_prime, (c * U - I) * q_prime]).transpose()
    prime_mat = matrix([p_prime, q_prime]).transpose()
    prime_mat_inv = prime_mat.inverse()
    V_v = (p - q).norm() * F * M * prime_mat_inv

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
    assert not is_in_sp(q, p), "q must not be in sp(p)"

    P = V_from_V(p, q, p)
    Q = V_from_V(p, q, q)

    # calculate b the proj of z onto the line q + span{p - q}
    pq = p - q
    norm_pq = pq.norm()
    proj_coeff = pq.dot_product(z - q) / (norm_pq ** 2)
    b = q + proj_coeff * pq

    B = V_from_V(p, q, b)

    # v orthogonal to sp{p, q}
    v = p.cross_product(q)

    scale = -(z - b).norm() / norm_pq # -||z-b|| / ||p-q||
    
    # [v]_x
    v_cross = matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    U = scale * (v_cross * (P - Q))

    # u = ||z-b|| * v
    u = (z - b).norm() * v

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
    assert is_correct_dim(p, 3), "p must be in R^3"
    assert is_correct_dim(q, 3), "q must be in R^3"
    assert is_correct_dim(z, 3), "z must be in R^3"
    assert is_correct_dim(l, 3), "l must be in R^3"
    assert is_nonzero(p), "p must not be zero"
    assert is_nonzero(l), "l must not be zero"
    assert not is_in_sp(q, p), "q must not be in sp(p)"

    # norm l
    norm_l = l.norm()
    l = l / norm_l
    l1, l2, l3 = l[0], l[1], l[2]
    
    if abs(l1) < 1e-14 and abs(l2) < 1e-14: # (l1, l2) = (0,0)
        l_orth = vector([0.0, 1.0, 0.0]) # (0,1,0)^T
    else:
        denom = sqrt(l1 ** 2 + l2 ** 2) # ||(l1, l2)||
        l_orth = vector([-l2, l1, 0.0]) / denom # (-l2, l1, 0)^T /||(l2,l1)||

    l_cross = l.cross_product(l_orth) # l x l_orth
    L = matrix([l_orth, l_cross]).transpose() # [l_orth, l x l_orth] (3x2)

    P, Q, B, U, u = PQBU(p, q, z)

    # c = ||L^Tu||
    c_vec = L.transpose() * u
    c = c_vec.norm()

    # check if u is zero vector
    if u.norm() < 1e-14:
        H = identity_matrix(2)  # H = I (2x2)
    else:
        w = L.transpose() * u # L^T u
        H = rotate(w, vector([0.0, 1.0]))

    # F = H L^T B
    F = H * (L.transpose() * B)
    # G = H L^T U
    G = H * (L.transpose() * U)

    return F, G, c


def reduce_to_constrained_linear_regression_msolve(F, G, y=None, mspath="msolve"):
    """
    This finds the point on the ellipsoid closest to the origin using msolve
    """
    if y is None:
        y = vector([1.0, 0.0])
    
    R = PolynomialRing(QQ, ['x1', 'x2', 'lam'], order='degrevlex')
    x1, x2, lam = R.gens()
    
    # convert F, G to rational numbers
    F_rat = matrix(QQ, [[QQ(F[i][j]) for j in range(2)] for i in range(2)])
    G_rat = matrix(QQ, [[QQ(G[i][j]) for j in range(2)] for i in range(2)])
    y_rat = vector(QQ, [QQ(y[i]) for i in range(2)])
    
    # calc Fx - Gy
    Fx_minus_Gy = [
        F_rat[0][0]*x1 + F_rat[0][1]*x2 - (G_rat[0][0]*y_rat[0] + G_rat[0][1]*y_rat[1]),
        F_rat[1][0]*x1 + F_rat[1][1]*x2 - (G_rat[1][0]*y_rat[0] + G_rat[1][1]*y_rat[1])
    ]
    
    # F^T * (Fx - Gy)
    FT_times_diff = [
        F_rat[0][0]*Fx_minus_Gy[0] + F_rat[1][0]*Fx_minus_Gy[1],
        F_rat[0][1]*Fx_minus_Gy[0] + F_rat[1][1]*Fx_minus_Gy[1]
    ]
    
    # sys of equations
    eqs = [
        FT_times_diff[0] - lam*x1,  # F^T*(Fx-Gy) = lam*x component 1
        FT_times_diff[1] - lam*x2,  # F^T*(Fx-Gy) = lam*x component 2  
        x1^2 + x2^2 - 1             # ||x||^2 = 1 constraint
    ]
    
    try:
        solutions = MSolveRealRoots(eqs, mspath=mspath, v=0, p=0)
        # find min solution
        best_x = None
        best_val = float('inf')
        
        for sol in solutions:
            # sol is a list [x1_val, x2_val, lam_val]
            x_val = vector([float(sol[0]), float(sol[1])])
            
            # check constraint ||x|| = 1
            if abs(x_val.norm() - 1.0) > 1e-10:
                continue
            
            # calc obj value ||Fx - Gy||
            obj_val = (F * x_val - G * y).norm()
            
            if obj_val < best_val:
                best_val = obj_val
                best_x = x_val
        
        return best_x, best_val
        
    except Exception as e:
        print(f"Msolve failed: {e}")



def check_answer(F, G, optimal_val, y=None):
    if y is None:
        y = vector([1.0, 0.0])
    
    # create the ellipsoid points and check if the optimal point is the closest to the origin
    theta_values = [2 * pi * k / 300 for k in range(300)]
    unit_circle = [vector([cos(theta), sin(theta)]) for theta in theta_values]
    ellipse_points = [F * x - G * y for x in unit_circle]

    # Check if any other point is closer
    all_dists = [(point[0]**2 + point[1]**2).sqrt() for point in ellipse_points]
    min_dist = min(all_dists)
    is_correct = optimal_val <= min_dist + 1e-6

    return is_correct

def visualize_answer(F, G, x_optimal, min_distance=0, y=None):
    if y is None:
        y = vector([1.0, 0.0])
    
    theta_values = [2 * pi * k / 300 for k in range(300)]
    unit_circle = [vector([cos(theta), sin(theta)]) for theta in theta_values]
    ellipse_points = [F * x - G * y for x in unit_circle]
    optimal_point = F * x_optimal - G * y
    
    # Convert to lists for matplotlib
    ellipse_x = [float(point[0]) for point in ellipse_points]
    ellipse_y = [float(point[1]) for point in ellipse_points]
    opt_x, opt_y = float(optimal_point[0]), float(optimal_point[1])
    
    plt.figure(figsize=(6, 6))
    plt.plot(ellipse_x, ellipse_y)
    plt.scatter(0, 0, color='black', label="Origin")
    plt.scatter(opt_x, opt_y, color='red', label="Optimal Point")
    plt.plot([0, opt_x], [0, opt_y], color='gray', linestyle='--')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def run_tests():
    pqzl = [
        (vector([1.0, 0.0, 0.0]), vector([0.0, 1.0, 0.0]), vector([0.5, 0.5, 1.0]), vector([1.0, 1.0, 1.0])),
        (vector([2.0, 0.0, 0.0]), vector([0.0, 3.0, 0.0]), vector([1.0, 1.0, 2.0]), vector([-1.0, -1.0, -2.0])),
        (vector([1.0, 0.0, 0.0]), vector([0.0, 1.0, 0.0]), vector([0.5, 1.0, 2]), vector([-1.0, 1.0, 1.0])),
    ]
    
    for i, (p, q, z, l) in enumerate(pqzl):
        F, G, _ = FG(p, q, z, l)
        x_opt, val = reduce_to_constrained_linear_regression_msolve(F, G)
        is_correct = check_answer(F, G, val)
        
        if is_correct:
            print("Optimal x:", x_opt)
            print("Minimum distance:", val)
        
        visualize_answer(F, G, x_opt, val)
    

# Main execution
if __name__ == "__main__":
    run_tests()