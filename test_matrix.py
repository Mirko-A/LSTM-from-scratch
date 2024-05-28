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

    def test_full(self):
        m = Matrix([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        value = 3.0
        expected = Matrix([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
        self.assertEqual(Matrix.full_like(m, value), expected)

    def test_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        expected = Matrix([[1, 4], [2, 5], [3, 6]])
        self.assertEqual(m.transpose(), expected)

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

    def test_exp(self):
        m = Matrix([[1, 2], [3, 4]])
        expected = Matrix([[2.718281828459045, 7.38905609893065], [20.085536923187668, 54.598150033144236]])
        self.assertEqual(m.exp(), expected)

    def test_log(self):
        m = Matrix([[2.718281828459045, 7.38905609893065], [20.08553692, 54.59815003]])
        expected = Matrix([[1.0, 2.0], [2.9999999998412954, 3.9999999999424114]])
        self.assertEqual(m.log(), expected)

if __name__ == '__main__':
    unittest.main()