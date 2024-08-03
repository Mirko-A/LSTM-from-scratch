from __future__ import annotations
from typing import Optional
import math
import random

class Matrix:
    def __init__(self, data: list[list[float]]) -> None:
        assert data, \
            "ERROR: Matrix must have at least one row."

        self.row_n = len(data)
        self.col_n = len(data[0])

        assert all(row and len(row) == self.col_n for row in data), \
            "ERROR: All matrix rows must have the same (non-zero) length."

        self.data = data

    @staticmethod
    def full(row_n: int, col_n: int, value: float) -> Matrix:
        data: list[list[float]] = []

        for _ in range(row_n):
            row: list[float] = []

            for _ in range(col_n):
                row.append(value)

            data.append(row)

        return Matrix(data)

    @staticmethod
    def zeros(row_n: int, col_n: int) -> Matrix:
        return Matrix.full(row_n, col_n, 0.0)
    
    @staticmethod
    def ones(row_n: int, col_n: int) -> Matrix:
        return Matrix.full(row_n, col_n, 1.0)

    @staticmethod
    def full_like(other: Matrix, value: float) -> Matrix:
        return Matrix.full(other.row_n, other.col_n, value)

    @staticmethod
    def zeros_like(other: Matrix) -> Matrix:
        return Matrix.full_like(other, 0.0)
    
    @staticmethod
    def ones_like(other: Matrix) -> Matrix:
        return Matrix.full_like(other, 1.0)

    @staticmethod
    def uniform(row_n: int, col_n: int, low: float, high: float) -> Matrix:
        data: list[list[float]] = []

        for _ in range(row_n):
            row: list[float] = []

            for _ in range(col_n):
                row.append(random.uniform(low, high))

            data.append(row)

        return Matrix(data)

    @property
    def T(self) -> Matrix:
        return self.transpose()

    def transpose(self) -> Matrix:
        data: list[list[float]] = []

        for col_i in range(self.col_n):
            row: list[float] = []

            for row_i in range(self.row_n):
                row.append(self.data[row_i][col_i])

            data.append(row)

        return Matrix(data)

    def expand(self, axis: int, new_size: int) -> Matrix:
        if axis == 0:
            assert self.row_n == 1, "ERROR: Axis 0 can only expand row vectors."

            data: list[list[float]] = []

            for _ in range(new_size):
                row: list[float] = []

                for col_i in range(self.col_n):
                    row.append(self.data[0][col_i])

                data.append(row)

            return Matrix(data)
        elif axis == 1:
            assert self.col_n == 1, "ERROR: Axis 1 can only expand column vectors."
            
            data: list[list[float]] = []

            for row_i in range(self.row_n):
                row: list[float] = []

                for _ in range(new_size):
                    row.append(self.data[row_i][0])

                data.append(row)

            return Matrix(data)
        else:
            assert False, "ERROR: Invalid axis value."

    def neg(self) -> Matrix:
        return -self

    def add(self, other: Matrix) -> Matrix:
        return self + other
    
    def sub(self, other: Matrix) -> Matrix:
        return self - other
    
    def mul(self, other: Matrix) -> Matrix:
        return self * other
    
    def div(self, other: Matrix) -> Matrix:
        return self / other
    
    def matmul(self, other: Matrix) -> Matrix:
        return self @ other
    
    def pow(self, other: Matrix) -> Matrix:
        return self ** other

    def sqrt(self) -> Matrix:
        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(math.sqrt(self.data[row_i][col_i]))

            data.append(row)

        return Matrix(data)

    def exp(self) -> Matrix:
        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(math.exp(self.data[row_i][col_i]))

            data.append(row)

        return Matrix(data)

    def log(self) -> Matrix:
        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(math.log(self.data[row_i][col_i]))

            data.append(row)

        return Matrix(data)
    
    def tanh(self) -> Matrix:
        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(math.tanh(self.data[row_i][col_i]))

            data.append(row)

        return Matrix(data)

    def sigmoid(self) -> Matrix:
        ones = Matrix.ones_like(self)
        return ones / (ones + (-self).exp())

    def sum(self, axis: Optional[int] = None) -> Matrix:
        if axis is None:
            val = 0.0

            for row_i in range(self.row_n):
                for col_i in range(self.col_n):
                    val += self.data[row_i][col_i]

            return Matrix([[val]])
        elif axis == 0:
            data: list[float] = []

            for col_i in range(self.col_n):
                val = 0.0

                for row_i in range(self.row_n):
                    val += self.data[row_i][col_i]

                data.append(val)

            return Matrix([data])
        elif axis == 1:
            data: list[list[float]] = []

            for row_i in range(self.row_n):
                val = 0.0

                for col_i in range(self.col_n):
                    val += self.data[row_i][col_i]

                data.append([val])

            return Matrix(data)
        else:
            assert False, "ERROR: Invalid axis value."

    def softmax(self, axis: Optional[int]) -> Matrix:
        exp = self.exp()

        if axis == 0:
            return exp / exp.sum(0).expand(0, self.row_n)
        elif axis == 1:
            return exp / exp.sum(1).expand(1, self.col_n)
        else:
            assert False, "ERROR: Invalid axis value."

    def equal(self, other: Matrix) -> bool:
        return self == other
    
    def all_close(self, other: Matrix, tolerance: float = 1e-5) -> bool:
        assert self._dims_match_with(other), \
            "ERROR: Matrices must have the same dimensions."

        for row_i in range(self.row_n):
            for col_i in range(self.col_n):
                if abs(self.data[row_i][col_i] - other.data[row_i][col_i]) > tolerance:
                    return False

        return True

    def at(self, row_i: int, col_i: int) -> float:
        return self.data[row_i][col_i]

    def scalar(self) -> float:
        assert self._is_scalar(), "ERROR: Matrix must be a scalar."

        return self.data[0][0]

    def __neg__(self) -> Matrix:
        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(-self.data[row_i][col_i])

            data.append(row)

        return Matrix(data)

    def __add__(self, other: Matrix) -> Matrix:
        assert self._dims_match_with(other), \
            "ERROR: Matrices must have the same dimensions."

        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(self.data[row_i][col_i] + other.data[row_i][col_i])

            data.append(row)

        return Matrix(data)
    
    def __sub__(self, other: Matrix) -> Matrix:
        assert self._dims_match_with(other), \
            "ERROR: Matrices must have the same dimensions."

        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(self.data[row_i][col_i] - other.data[row_i][col_i])

            data.append(row)

        return Matrix(data)
    
    def __mul__(self, other: Matrix) -> Matrix:
        assert self._dims_match_with(other), \
            "ERROR: Matrices must have the same dimensions."

        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(self.data[row_i][col_i] * other.data[row_i][col_i])

            data.append(row)

        return Matrix(data)


    def __truediv__(self, other: Matrix) -> Matrix:
        assert self._dims_match_with(other), \
            "ERROR: Matrices must have the same dimensions."

        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(self.data[row_i][col_i] / other.data[row_i][col_i])

            data.append(row)

        return Matrix(data)

    def __pow__(self, other: Matrix) -> Matrix:
        assert self._dims_match_with(other), \
            "ERROR: Matrices must have the same dimensions."

        data: list[list[float]] = []

        for row_i in range(self.row_n):
            row: list[float] = []

            for col_i in range(self.col_n):
                row.append(self.data[row_i][col_i] ** other.data[row_i][col_i])

            data.append(row)

        return Matrix(data)

    def __matmul__(self, other: Matrix) -> Matrix:
        assert self.col_n == other.row_n, \
            "ERROR: Number of columns in the first matrix must match the number of rows in the second matrix."

        other = other.T

        data: list[list[float]] = []

        for x_row_i in range(self.row_n):
            row: list[float] = []

            for y_row_i in range(other.row_n):
                val = 0.0

                for col_i in range(self.col_n):
                    val += self.data[x_row_i][col_i] * other.data[y_row_i][col_i]

                row.append(val)

            data.append(row)

        return Matrix(data)

    def __eq__(self, other: Matrix) -> bool:
        for row_i in range(self.row_n):
            for col_i in range(self.col_n):
                if self.data[row_i][col_i] != other.data[row_i][col_i]:
                    return False

        return True

    def __repr__(self) -> str:
        return f"Matrix({self.data})"
    
    def _dims_match_with(self, other: Matrix) -> bool:
        return self.row_n == other.row_n and self.col_n == other.col_n
    
    def _is_scalar(self) -> bool:
        return self.row_n == 1 and self.col_n == 1