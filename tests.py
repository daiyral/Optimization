import unittest
import numpy as np
from algorithms import rotate, V_from_V, PQBU, FG


class TestAlgorithms(unittest.TestCase):

    def test_identity_rotation(self):
        w = np.array([1.0, 0.0])
        u = np.array([1.0, 0.0])
        U = rotate(w, u)
        np.testing.assert_allclose(U, np.eye(2), atol=1e-6)

    def test_90_degree_ccw(self):
        w = np.array([1.0, 0.0])
        u = np.array([0.0, 1.0])
        U = rotate(w, u)
        expected = np.array([[0, -1],
                             [1, 0]])
        np.testing.assert_allclose(U, expected, atol=1e-6)

    def test_180_degree(self):
        w = np.array([1.0, 0.0])
        u = np.array([-1.0, 0.0])
        U = rotate(w, u)
        expected = np.array([[-1, 0],
                             [0, -1]])
        np.testing.assert_allclose(U, expected, atol=1e-6)

    def compute_gR(self, p, q, R):
        p_norm = p / np.linalg.norm(p)
        q_proj = np.linalg.norm(p) ** 2 * q - np.dot(q, p) * p
        q_proj_norm = np.linalg.norm(q_proj)

        if np.isclose(q_proj_norm, 0):
            F = np.column_stack([p_norm, np.zeros_like(p_norm)])
        else:
            F = np.column_stack([p_norm, q_proj / q_proj_norm])

        diff = (p - q) / np.linalg.norm(p - q)
        return F.T @ (R @ diff)

    def test_lemma3_part_i(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        R = np.array([[0, -1], [1, 0]])  # 90-degree rotation

        gR = self.compute_gR(p, q, R)

        Vp = V_from_V(p, q, p)
        Vq = V_from_V(p, q, q)

        t1 = Vp @ gR - R @ p
        t2 = Vq @ gR - R @ q

        np.testing.assert_allclose(t1, t2, atol=1e-10)

    def test_lemma3_part_ii(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        v = np.array([0.5, 0.5])
        R = np.array([[0, -1], [1, 0]])  # 90-degree rotation

        gR = self.compute_gR(p, q, R)

        Vp = V_from_V(p, q, p)
        Vv = V_from_V(p, q, v)

        t_expected = Vp @ gR - R @ p
        t_actual = Vv @ gR - R @ v

        np.testing.assert_allclose(t_actual, t_expected, atol=1e-10)

    def test_PQBU_projection(self):
        # Test that b is the projection of z onto the line q + span{p-q}
        p = np.array([1.0, 2.0, 3.0])
        q = np.array([4.0, 5.0, 6.0])
        z = np.array([7.0, 8.0, 9.0])
        # Expected: b = q + ((p-q)(p-q)^T/(||p-q||^2)) (z-q)
        pq = p - q
        norm_pq = np.linalg.norm(pq)
        proj_mat = np.outer(pq, pq) / (norm_pq ** 2)
        b_expected = q + proj_mat @ (z - q)

        # Run PQBU
        P, Q, B, U, u = PQBU(p, q, z)

        # Since B = V_from_V(p,q,b) and b is computed internally,
        # we indirectly check that the computed b is consistent.
        # We check that u is scaled by ||z-b|| and that its direction is p x q.
        v = np.cross(p, q)
        self.assertTrue(np.allclose(u, np.linalg.norm(z - b_expected) * v, atol=1e-6))

        # Also check shapes
        self.assertEqual(P.shape, (3, 2))
        self.assertEqual(Q.shape, (3, 2))
        self.assertEqual(B.shape, (3, 2))
        self.assertEqual(U.shape, (3, 2))
        self.assertEqual(u.shape, (3,))

    def test_PQBU_U_property(self):
        # Test the specific structure of U from Algorithm 3:
        # U should be equal to - (||z - b||/||p - q||) * ([v]_x (P - Q))
        p = np.array([2.0, 1.0, 3.0])
        q = np.array([1.0, 4.0, 0.0])
        z = np.array([3.0, 2.0, 5.0])
        # Compute pq, norm_pq, v as in PQBU:
        pq = p - q
        norm_pq = np.linalg.norm(pq)
        v = np.cross(p, q)
        b = q + (np.outer(pq, pq) / (norm_pq ** 2)) @ (z - q)
        scale = - (np.linalg.norm(z - b) / norm_pq)

        # Build cross product matrix of v:
        v_cross = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ])

        # Run PQBU:
        P, Q, B_out, U, u = PQBU(p, q, z)

        U_expected = scale * (v_cross @ (P - Q))
        np.testing.assert_allclose(U, U_expected, atol=1e-6)

    def test_FG_properties(self):
        # Test FG to verify that c equals ||L^T u|| and F, G are 2x2.
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        z = np.array([0.5, 0.5, 1.0])
        l = np.array([0.0, 1.0, 0.0])
        # Run FG:
        F, G, c = FG(p, q, z, l)

        # Check shapes of F and G are 2x2, and c is scalar.
        self.assertEqual(F.shape, (2, 2))
        self.assertEqual(G.shape, (2, 2))
        self.assertTrue(np.isscalar(c) or (isinstance(c, np.ndarray) and c.shape == ()))

        # Recompute L and u as in FG to check c = ||L^T u||
        norm_l = np.linalg.norm(l)
        l_normalized = l / norm_l
        l1, l2, l3 = l_normalized
        if abs(l1) < 1e-14 and abs(l2) < 1e-14:
            l_orth = np.array([0.0, 1.0, 0.0])
        else:
            denom = np.sqrt(l1 ** 2 + l2 ** 2)
            l_orth = np.array([-l2, l1, 0.0]) / denom
        l_cross = np.cross(l_normalized, l_orth)
        L = np.column_stack((l_orth, l_cross))

        # Get u from PQBU(p,q,z)
        _, _, _, _, u = PQBU(p, q, z)
        c_expected = np.linalg.norm(L.T @ u)
        np.testing.assert_allclose(c, c_expected, atol=1e-6)

    def test_FG_rotation(self):
        # Test that if u is near zero then H is identity and thus F ~ L^T B, G ~ L^T U.
        p = np.array([3.0, 0.0, 0.0])
        q = np.array([0.0, 3.0, 0.0])
        # Choose z such that its projection is almost exactly at q:
        z = q.copy() + 1e-9 * np.array([1.0, 0.0, 0.0])
        l = np.array([0.0, 1.0, 0.0])
        F, G, c = FG(p, q, z, l)
        # In this case u should be near zero, hence the rotation H is the identity.
        # Therefore, F should approximately equal L^T B and G approximately L^T U.
        # We recompute L and then compare.
        norm_l = np.linalg.norm(l)
        l_normalized = l / norm_l
        l1, l2, l3 = l_normalized
        if abs(l1) < 1e-14 and abs(l2) < 1e-14:
            l_orth = np.array([0.0, 1.0, 0.0])
        else:
            denom = np.sqrt(l1 ** 2 + l2 ** 2)
            l_orth = np.array([-l2, l1, 0.0]) / denom
        l_cross = np.cross(l_normalized, l_orth)
        L = np.column_stack((l_orth, l_cross))
        _, _, B, U, u = PQBU(p, q, z)
        B2 = L.T @ B
        U2 = L.T @ U

        # Since u is nearly zero, H should be I.
        np.testing.assert_allclose(F, B2, atol=1e-6)
        np.testing.assert_allclose(G, U2, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
