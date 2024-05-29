import unittest
from matrix import Matrix

class MatrixTests(unittest.TestCase):
    def test_init(self):
        data = [[1, 2, 3], [4, 5, 6]]
        m = Matrix(data)
        self.assertEqual(m.data, data)
        self.assertEqual(m.row_n, 2)
        self.assertEqual(m.col_n, 3)

        invalid_data = [[]]
        with self.assertRaises(AssertionError):
            Matrix(invalid_data)

        invalid_data = [[1, 2], [3]]
        with self.assertRaises(AssertionError):
            Matrix(invalid_data)
            
    def test_full(self):
        row_n = 2
        col_n = 3
        value = 5.0
        expected = Matrix([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        self.assertEqual(Matrix.full(row_n, col_n, value), expected)

    def test_full_like(self):
        m = Matrix([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        value = 3.0
        expected = Matrix([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
        self.assertEqual(Matrix.full_like(m, value), expected)

    def test_uniform(self):
        row_n = 2
        col_n = 3
        low = -1.0
        high = 1.0
        m = Matrix.uniform(row_n, col_n, low, high)
        self.assertEqual(m.row_n, row_n)
        self.assertEqual(m.col_n, col_n)

        for row in m.data:
            for value in row:
                self.assertTrue(low <= value <= high)

    def test_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        expected = Matrix([[1, 4], [2, 5], [3, 6]])
        self.assertEqual(m.transpose(), expected)

    def test_expand(self):
        m = Matrix([[1], [3]])
        axis = 1
        new_size = 3
        expected = Matrix([[1, 1, 1], [3, 3, 3]])
        self.assertEqual(m.expand(axis, new_size), expected)
        
        m = Matrix([[1, 3]])
        axis = 0
        new_size = 3
        expected = Matrix([[1, 3], [1, 3], [1, 3]])
        self.assertEqual(m.expand(axis, new_size), expected)

    def test_neg(self):
        m = Matrix([[1, 2], [3, 4]])
        expected = Matrix([[-1, -2], [-3, -4]])
        self.assertEqual(m.neg(), expected)

        m = Matrix([[0, 0], [0, 0]])
        expected = Matrix([[0, 0], [0, 0]])
        self.assertEqual(m.neg(), expected)

        m = Matrix([[-1, -2], [3, 4]])
        expected = Matrix([[1, 2], [-3, -4]])
        self.assertEqual(m.neg(), expected)

    def test_add(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        expected = Matrix([[6, 8], [10, 12]])
        self.assertEqual(m1.add(m2), expected)

    def test_sub(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        expected = Matrix([[-4, -4], [-4, -4]])
        self.assertEqual(m1.sub(m2), expected)

    def test_mul(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        expected = Matrix([[5, 12], [21, 32]])
        self.assertEqual(m1.mul(m2), expected)

    def test_div(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        expected = Matrix([[0.2, 0.3333333333333333], [0.42857142857142855, 0.5]])
        self.assertEqual(m1.div(m2), expected)

    def test_matmul(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        expected = Matrix([[19, 22], [43, 50]])
        self.assertEqual(m1.matmul(m2), expected)

    def test_pow(self):
        m = Matrix([[1, 2], [3, 4]])
        exponent = Matrix([[2, 2], [2, 2]])
        expected = Matrix([[1, 4], [9, 16]])
        self.assertEqual(m.pow(exponent), expected)

    def test_sqrt(self):
        m = Matrix([[4, 9], [16, 25]])
        expected = Matrix([[2.0, 3.0], [4.0, 5.0]])
        self.assertEqual(m.sqrt(), expected)

        m = Matrix([[1, 2], [3, 4]])
        expected = Matrix([[1.0, 1.4142135623730951], [1.7320508075688772, 2.0]])
        self.assertEqual(m.sqrt(), expected)

        m = Matrix([[0, 0], [0, 0]])
        expected = Matrix([[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(m.sqrt(), expected)

    def test_exp(self):
        m = Matrix([[1, 2], [3, 4]])
        expected = Matrix([[2.718281828459045, 7.38905609893065], [20.085536923187668, 54.598150033144236]])
        self.assertEqual(m.exp(), expected)

    def test_log(self):
        m = Matrix([[2.718281828459045, 7.38905609893065], [20.08553692, 54.59815003]])
        expected = Matrix([[1.0, 2.0], [2.9999999998412954, 3.9999999999424114]])
        self.assertEqual(m.log(), expected)

    def test_sigmoid(self):
        m = Matrix([[0, 1], [-1, 2]])
        expected = Matrix([[0.5, 0.7310585786300049], [0.2689414213699951, 0.8807970779778823]])
        self.assertEqual(m.sigmoid(), expected)

        m = Matrix([[0.5, -0.5], [1.0, -1.0]])
        expected = Matrix([[0.6224593312018546, 0.3775406687981454], [0.7310585786300049, 0.2689414213699951]])
        self.assertEqual(m.sigmoid(), expected)

        m = Matrix([[0, 0], [0, 0]])
        expected = Matrix([[0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(m.sigmoid(), expected)

    def test_tanh(self):
        m = Matrix([[0, 1], [-1, 2]])
        expected = Matrix([[0.0, 0.7615941559557649], [-0.7615941559557649, 0.9640275800758169]])
        self.assertEqual(m.tanh(), expected)

        m = Matrix([[0.5, -0.5], [1.0, -1.0]])
        expected = Matrix([[0.46211715726000974, -0.46211715726000974], [0.7615941559557649, -0.7615941559557649]])
        self.assertEqual(m.tanh(), expected)

        m = Matrix([[0, 0], [0, 0]])
        expected = Matrix([[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(m.tanh(), expected)

    def test_sum(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        expected = Matrix([[21]])
        self.assertEqual(m.sum(), expected)

        m = Matrix([[1, 2, 3], [4, 5, 6]])
        expected = Matrix([[5, 7, 9]])
        self.assertEqual(m.sum(axis=0), expected)

        m = Matrix([[1, 2, 3], [4, 5, 6]])
        expected = Matrix([[6], [15]])
        self.assertEqual(m.sum(axis=1), expected)

    def test_softmax(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        expected = Matrix([[0.04742587317756678, 0.04742587317756678, 0.04742587317756678], [0.9525741268224331, 0.9525741268224331, 0.9525741268224331]])
        self.assertTrue(m.softmax(axis=0).all_close(expected, tolerance=1e-7))

        m = Matrix([[1, 2, 3], [4, 5, 6]])
        expected = Matrix([[0.09003057317038046, 0.24472847105479764, 0.6652409557748219], [0.09003057317038046, 0.24472847105479764, 0.6652409557748219]])
        self.assertTrue(m.softmax(axis=1).all_close(expected, tolerance=1e-7))

    def test_at(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        expected = 5
        self.assertEqual(m.at(1, 1), expected)

    def test_scalar(self):
        m = Matrix([[1.0]])
        expected = 1.0
        self.assertEqual(m.scalar(), expected)

if __name__ == '__main__':
    unittest.main()