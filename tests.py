import unittest
import numpy as np
from algorithms import rotate, V_from_V, PQBU, FG

# Danny Aibinder: 318239639
# Bradley Feitsvaig: 311183073

class TestAlgorithms(unittest.TestCase):

    def test_identity_rotation(self):
        # w,u dont need rotation so we expect the get I matrix as a result
        w = np.array([1.0, 0.0])
        u = np.array([1.0, 0.0])
        U = rotate(w, u)
        np.testing.assert_allclose(U, np.eye(2))

    def test_90_degree_ccw(self):
        w = np.array([1.0, 0.0])
        u = np.array([0.0, 1.0])
        U = rotate(w, u)
        expected = np.array([[0, -1],
                             [1, 0]]) # 90 degree ccw matrix expected
        np.testing.assert_allclose(U, expected)

    def test_180_degree(self):
        w = np.array([1.0, 0.0])
        u = np.array([-1.0, 0.0])
        U = rotate(w, u)
        expected = np.array([[-1, 0],
                             [0, -1]]) # 180 rotation matrix expected
        np.testing.assert_allclose(U, expected)

    def compute_gR(self, p, q, R):
        # g(R) = F(p,q)^T R (p-q)/||p-q||
        # calculate F
        p_norm = p / np.linalg.norm(p)
        pq = (np.linalg.norm(p) ** 2) * q - (q.T @ p) * p
        pq_norm = pq / np.linalg.norm(pq)
        F = np.column_stack((p_norm, pq_norm))  # (d,2)

        diff = (p - q) / np.linalg.norm(p - q)
        return F.T @ (R @ diff)

    def test_lemma3(self):
        # test that V(p)g(R)-Rp = V(q)g(R) - Rq
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        R = np.array([[0, -1], [1, 0]]) # 90-degree rotation

        gR = self.compute_gR(p, q, R)

        Vp = V_from_V(p, q, p)
        Vq = V_from_V(p, q, q)

        t1 = Vp @ gR - R @ p # V(p)g(R)-Rp
        t2 = Vq @ gR - R @ q # V(q)g(R) - Rq

        np.testing.assert_allclose(t1, t2)

    def test_theorem_16(self):
        # tests if z can be found in the unit circle from the matrix B,U and vector u from PQBU 
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        z = np.array([0.5, 0.5, 1.0])
        P, Q, B, U, u = PQBU(p, q, z)
        found = False
        # search for x, y on the unit circle thatreconstructs z
        for theta in np.linspace(0, 2*np.pi, 300):
            x = np.array([np.cos(theta), np.sin(theta)]) # point 
            for phi in np.linspace(0, 2*np.pi, 300):
                y = np.array([np.cos(phi), np.sin(phi)]) # point
                candidate = (B + y[0] * U) @ x + y[1] * u #  (B+y_1U)x + y_2u
                if np.allclose(candidate, z, atol=1e-2):
                    found = True
                    break
            if found:
                break
        
        self.assertTrue(found, "z not found in circle")

    def get_L(self,l):
        # implemented in algorithsm.py
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
        return L

    def test_corollary_17(self):
        # we test that ||L^T((B+y_1U)x + y_2u)|| = ||(F+y_1G)x + (0,y_2c)^T||
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        z = np.array([0.5, 0.5, 1.0])
        l = np.array([1.0, 1.0, 1.0])

        F, G, c = FG(p, q, z, l)
        P, Q, B, U, u = PQBU(p, q, z)

        L = self.get_L(l)

        # circle points
        angles = np.linspace(0, 2*np.pi, 100)
        for theta in angles:
            for phi in angles:
                x = np.array([np.cos(theta), np.sin(theta)])
                y = np.array([np.cos(phi), np.sin(phi)])

                # ||L^T((B + y1 U)x + y2 u)||
                left = (B + y[0] * U) @ x + y[1] * u
                left_norm = np.linalg.norm(L.T @ left)

                # ||(F + y1 G)x + (0, c y2)^T||
                right = (F + y[0] * G) @ x + np.array([0.0, c * y[1]])
                right_norm = np.linalg.norm(right)

                np.testing.assert_allclose(left_norm, right_norm)


if __name__ == '__main__':
    unittest.main()
