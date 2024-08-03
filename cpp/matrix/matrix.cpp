#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

#include "matrix.hpp"

Matrix::Matrix(std::vector<std::vector<float>> data)
    : row_n(data.size()), col_n(data[0].size()), data(std::move(data)) {
    assert(!this->data.empty());

    size_t col_size = this->data[0].size();
    for (const auto &row : this->data) {
        assert(row.size() == col_size);
    }
}

Matrix Matrix::full(uint32_t row_n, uint32_t col_n, float value) {
    return Matrix(std::vector<std::vector<float>>(row_n, std::vector<float>(col_n, value)));
}

Matrix Matrix::zeros(uint32_t row_n, uint32_t col_n) {
    return full(row_n, col_n, 0.0f);
}

Matrix Matrix::ones(uint32_t row_n, uint32_t col_n) {
    return full(row_n, col_n, 1.0f);
}

Matrix Matrix::full_like(const Matrix &other, float value) {
    return full(other.data.size(), other.data[0].size(), value);
}

Matrix Matrix::zeros_like(const Matrix &other) {
    return full_like(other, 0.0f);
}

Matrix Matrix::ones_like(const Matrix &other) {
    return full_like(other, 1.0f);
}

Matrix Matrix::arange(uint32_t row_n, uint32_t col_n, uint32_t start) {
    std::vector<std::vector<float>> data(row_n, std::vector<float>(col_n));

    float val = static_cast<float>(start);
    for (auto &row : data) {
        std::generate(row.begin(), row.end(),
                      [&]() { return val++; });
    }

    return Matrix(data);
}

Matrix Matrix::uniform(uint32_t row_n, uint32_t col_n, float low, float high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(low, high);

    std::vector<std::vector<float>> data(row_n, std::vector<float>(col_n));

    for (auto &row : data) {
        std::generate(row.begin(), row.end(),
                      [&]() { return dis(gen); });
    }

    return Matrix(data);
}

Matrix Matrix::transpose() const {
    std::vector<std::vector<float>> transposed(col_n, std::vector<float>(row_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            transposed[j][i] = data[i][j];
        }
    }

    return Matrix(transposed);
}

Matrix Matrix::T() const {
    return transpose();
}

Matrix Matrix::expand(int axis, uint32_t new_size) const {
    assert(axis == 0 || axis == 1);

    if (axis == 0) {
        assert(row_n == 1);

        std::vector<std::vector<float>> expanded(new_size, std::vector<float>(col_n));
        for (uint32_t i = 0; i < new_size; ++i) {
            for (uint32_t j = 0; j < col_n; ++j) {
                expanded[i][j] = data[0][j];
            }
        }

        return Matrix(expanded);
    } else {
        assert(col_n == 1);

        std::vector<std::vector<float>> expanded(row_n, std::vector<float>(new_size));
        for (uint32_t i = 0; i < row_n; ++i) {
            for (uint32_t j = 0; j < new_size; ++j) {
                expanded[i][j] = data[i][0];
            }
        }

        return Matrix(expanded);
    }
}

Matrix Matrix::neg() const {
    std::vector<std::vector<float>> neg_data = data;

    for (auto &row : neg_data) {
        for (auto &elem : row) {
            elem = -elem;
        }
    }
    return Matrix(neg_data);
}

Matrix Matrix::add(const Matrix &other) const {
    assert(dims_same_as(other));

    std::vector<std::vector<float>> result(row_n, std::vector<float>(col_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            result[i][j] = data[i][j] + other.data[i][j];
        }
    }

    return Matrix(result);
}

Matrix Matrix::sub(const Matrix &other) const {
    assert(dims_same_as(other));

    std::vector<std::vector<float>> result(row_n, std::vector<float>(col_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            result[i][j] = data[i][j] - other.data[i][j];
        }
    }

    return Matrix(result);
}

Matrix Matrix::multiply(const Matrix &other) const {
    assert(dims_same_as(other));

    std::vector<std::vector<float>> result(row_n, std::vector<float>(col_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            result[i][j] = data[i][j] * other.data[i][j];
        }
    }

    return Matrix(result);
}

Matrix Matrix::divide(const Matrix &other) const {
    assert(dims_same_as(other));

    std::vector<std::vector<float>> result(row_n, std::vector<float>(col_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            result[i][j] = this->data[i][j] / other.data[i][j];
        }
    }

    return Matrix(result);
}

Matrix Matrix::pow(const Matrix &other) const {
    assert(dims_same_as(other));

    std::vector<std::vector<float>> result(row_n, std::vector<float>(col_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            result[i][j] = std::pow(data[i][j], other.data[i][j]);
        }
    }

    return Matrix(result);
}

Matrix Matrix::add(float scalar) const {
    return add(full_like(*this, scalar));
}

Matrix Matrix::sub(float scalar) const {
    return add(full_like(*this, -scalar));
}

Matrix Matrix::multiply(float scalar) const {
    return multiply(full_like(*this, scalar));
}

Matrix Matrix::divide(float scalar) const {
    return divide(full_like(*this, scalar));
}

Matrix Matrix::pow(float other) const {
    return pow(full_like(*this, other));
}

Matrix Matrix::operator-() const {
    return neg();
}

Matrix Matrix::operator+(const Matrix &other) const {
    return add(other);
}

Matrix Matrix::operator-(const Matrix &other) const {
    return sub(other);
}

Matrix Matrix::operator*(const Matrix &other) const {
    return multiply(other);
}

Matrix Matrix::operator/(const Matrix &other) const {
    return divide(other);
}

Matrix Matrix::operator+(float scalar) const {
    return add(full_like(*this, scalar));
}

Matrix Matrix::operator-(float scalar) const {
    return sub(full_like(*this, scalar));
}

Matrix Matrix::operator*(float scalar) const {
    return multiply(full_like(*this, scalar));
}

Matrix Matrix::operator/(float scalar) const {
    return divide(full_like(*this, scalar));
}

Matrix operator+(float scalar, const Matrix &matrix) {
    return matrix.add(scalar);
}

Matrix operator-(float scalar, const Matrix &matrix) {
    return Matrix::full_like(matrix, scalar).sub(matrix);
}

Matrix operator*(float scalar, const Matrix &matrix) {
    return matrix.multiply(scalar);
}

Matrix operator/(float scalar, const Matrix &matrix) {
    return Matrix::full_like(matrix, scalar).divide(matrix);
}

Matrix Matrix::matmul(const Matrix &other) const {
    assert(inner_dim_same_as(other));

    uint32_t new_row_n = row_n;
    uint32_t new_col_n = other.col_n;
    std::vector<std::vector<float>> result(new_row_n, std::vector<float>(new_col_n));

    for (uint32_t i = 0; i < new_row_n; ++i) {
        for (uint32_t j = 0; j < new_col_n; ++j) {
            float val = 0.0;
            for (uint32_t k = 0; k < col_n; ++k) {
                val += data[i][k] * other.data[k][j];
            }
            result[i][j] = val;
        }
    }

    return Matrix(result);
}

Matrix Matrix::sqrt() const {
    std::vector<std::vector<float>> result(row_n, std::vector<float>(col_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            result[i][j] = std::sqrt(data[i][j]);
        }
    }

    return Matrix(result);
}

Matrix Matrix::exp() const {
    std::vector<std::vector<float>> result(row_n, std::vector<float>(col_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            result[i][j] = std::exp(data[i][j]);
        }
    }

    return Matrix(result);
}

Matrix Matrix::log() const {
    std::vector<std::vector<float>> result(row_n, std::vector<float>(col_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            result[i][j] = std::log(data[i][j]);
        }
    }

    return Matrix(result);
}

Matrix Matrix::tanh() const {
    std::vector<std::vector<float>> result(row_n, std::vector<float>(col_n));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            result[i][j] = std::tanh(data[i][j]);
        }
    }

    return Matrix(result);
}

Matrix Matrix::sigmoid() const {
    Matrix ones = ones_like(*this);
    return ones.divide(ones.add((this->neg()).exp()));
}

Matrix Matrix::sum(std::optional<uint8_t> axis) const {
    if (axis.has_value()) {
        uint32_t ax = axis.value();

        assert(ax == 0 || ax == 1);

        if (ax == 0) {
            std::vector<std::vector<float>> result(1, std::vector<float>(col_n, 0.0f));
            for (uint32_t i = 0; i < row_n; ++i) {
                for (uint32_t j = 0; j < col_n; ++j) {
                    result[0][j] += data[i][j];
                }
            }

            return Matrix(result);
        } else {
            std::vector<std::vector<float>> result(row_n, std::vector<float>(1, 0.0f));
            for (uint32_t i = 0; i < row_n; ++i) {
                for (uint32_t j = 0; j < col_n; ++j) {
                    result[i][0] += data[i][j];
                }
            }

            return Matrix(result);
        }
    } else {
        float total_sum = 0.0f;

        for (uint32_t i = 0; i < row_n; ++i) {
            for (uint32_t j = 0; j < col_n; ++j) {
                total_sum += data[i][j];
            }
        }

        std::vector<std::vector<float>> data = {{total_sum}};
        return Matrix(data);
    }
}

Matrix Matrix::softmax(std::optional<uint8_t> axis) const {
    if (axis.has_value()) {
        uint32_t ax = axis.value();

        assert(ax == 0 || ax == 1);

        Matrix exp_data = exp();
        Matrix sum_exp = exp_data.sum(ax);
        uint32_t expanded_size = axis == 0 ? row_n : col_n;
        Matrix sum_exp_expanded = sum_exp.expand(ax, expanded_size);

        return exp_data.divide(sum_exp_expanded);
    } else {
        Matrix exp_data = exp();
        Matrix sum_exp = exp_data.sum(std::nullopt);
        Matrix sum_exp_expanded = sum_exp.expand(0, row_n).expand(1, col_n);

        return exp_data.divide(sum_exp_expanded);
    }
}

bool Matrix::equal(const Matrix &other) const {
    if (!dims_same_as(other)) {
        return false;
    }

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            if (data[i][j] != other.data[i][j]) {
                return false;
            }
        }
    }

    return true;
}

bool Matrix::all_close(const Matrix &other, float tolerance) const {
    assert(dims_same_as(other));

    for (uint32_t i = 0; i < row_n; ++i) {
        for (uint32_t j = 0; j < col_n; ++j) {
            if (std::abs(data[i][j] - other.data[i][j]) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

void Matrix::set(uint32_t row_i, uint32_t col_i, float value) {
    assert(row_i < row_n && col_i < col_n);
    data[row_i][col_i] = value;
}

float Matrix::at(uint32_t row_i, uint32_t col_i) const {
    assert(row_i < row_n && col_i < col_n);

    return data[row_i][col_i];
}

float Matrix::scalar() const {
    assert(row_n == 1 && col_n == 1);

    return data[0][0];
}

bool Matrix::dims_same_as(const Matrix &other) const {
    return row_n == other.row_n && col_n == other.col_n;
}

bool Matrix::inner_dim_same_as(const Matrix &other) const {
    return col_n == other.row_n;
}

void Matrix::print() const {
    std::cout << *this;
}

void Matrix::println() const {
    std::cout << *this;
    std::cout << std::endl;
}

std::ostream &operator<<(std::ostream &os, const Matrix &rhs) {
    os << "Matrix(" << rhs.row_n << ", " << rhs.col_n << ")\n";

    for (const auto &row : rhs.data) {
        for (const auto &elem : row) {
            os << elem << ' ';
        }
        os << '\n';
    }

    return os;
}