import unittest
import numpy as np
from algorithms import rotate, V_from_V


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


if __name__ == '__main__':
    unittest.main()
