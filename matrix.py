from __future__ import annotations
import math

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

        for row_i in range(row_n):
            row: list[float] = []

            for col_i in range(col_n):
                row.append(value)

            data.append(row)

        return Matrix(data)

    @staticmethod
    def full_like(other: Matrix, value: float) -> Matrix:
        return Matrix.full(other.row_n, other.col_n, value)

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