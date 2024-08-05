#include "matrix.hpp"
#include <gtest/gtest.h>

TEST(MatrixTest, DefaultConstructor) {
    Matrix m;
    Matrix expected;
    EXPECT_TRUE(m.equal(expected));
}

TEST(MatrixTest, Constructor) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    std::vector<std::vector<float>> expected_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m.equal(expected));
}

TEST(MatrixTest, CopyConstructor) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_copy(m);
    std::vector<std::vector<float>> expected_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_copy.equal(expected));
}

TEST(MatrixTest, MoveConstructor) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_move(std::move(m));
    std::vector<std::vector<float>> expected_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_move.equal(expected));
}

TEST(MatrixTest, CopyAssignment) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_copy = m;
    std::vector<std::vector<float>> expected_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_copy.equal(expected));
}

TEST(MatrixTest, MoveAssignment) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_move = std::move(m);
    std::vector<std::vector<float>> expected_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_move.equal(expected));
}

TEST(MatrixTest, Full) {
    Matrix m = Matrix::full(2, 2, 1);
    std::vector<std::vector<float>> expected_data = {{1.0f, 1.0f}, {1.0f, 1.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m.equal(expected));
}

TEST(MatrixTest, Zeros) {
    Matrix m = Matrix::zeros(2, 2);
    std::vector<std::vector<float>> expected_data = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m.equal(expected));
}

TEST(MatrixTest, Ones) {
    Matrix m = Matrix::ones(2, 2);
    std::vector<std::vector<float>> expected_data = {{1.0f, 1.0f}, {1.0f, 1.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m.equal(expected));
}

TEST(MatrixTest, FullLike) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_full_like = Matrix::full_like(m, 1);
    std::vector<std::vector<float>> expected_data = {{1.0f, 1.0f}, {1.0f, 1.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_full_like.equal(expected));
}

TEST(MatrixTest, ZerosLike) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_zeros_like = Matrix::zeros_like(m);
    std::vector<std::vector<float>> expected_data = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_zeros_like.equal(expected));
}

TEST(MatrixTest, OnesLike) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_ones_like = Matrix::ones_like(m);
    std::vector<std::vector<float>> expected_data = {{1.0f, 1.0f}, {1.0f, 1.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_ones_like.equal(expected));
}

TEST(MatrixTest, Arange) {
    Matrix m = Matrix::arange(2, 2);
    std::vector<std::vector<float>> expected_data = {{0.0f, 1.0f}, {2.0f, 3.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m.equal(expected));
}

Test(MatrixTest, ArangeFromStart) {
    Matrix m = Matrix::arange(2, 2, 1);
    std::vector<std::vector<float>> expected_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m.equal(expected));
}

TEST(MatrixTest, Uniform) {
    Matrix m = Matrix::uniform(2, 2, 0.0f, 1.0f);
    for (uint32_t i = 0; i < m.row_n(); ++i) {
        for (uint32_t j = 0; j < m.col_n(); ++j) {
            EXPECT_GE(m(i, j), 0.0f);
            EXPECT_LE(m(i, j), 1.0f);
        }
    }
}

TEST(MatrixTest, Transpose) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_transpose = m.transpose();
    std::vector<std::vector<float>> expected_data = {{1.0f, 3.0f}, {2.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_transpose.equal(expected));
}

TEST(MatrixTest, T) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_T = m.T();
    std::vector<std::vector<float>> expected_data = {{1.0f, 3.0f}, {2.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_T.equal(expected));
}

TEST(MatrixTest, Expand) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_expand = m.expand(0, 2);
    std::vector<std::vector<float>> expected_data = {{1.0f, 2.0f}, {1.0f, 2.0f}, {3.0f, 4.0f}, {3, 4}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_expand.equal(expected));
}

TEST(MatrixTest, Neg) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_neg = m.neg();
    std::vector<std::vector<float>> expected_data = {{-1.0f, -2.0f}, {-3.0f, -4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_neg.equal(expected));
}

TEST(MatrixTest, AddMatrix) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    Matrix m2(data2);
    Matrix m_add = m1.add(m2);
    std::vector<std::vector<float>> expected_data = {{6.0f, 8.0f}, {10.0f, 12.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_add.equal(expected));
}

TEST(MatrixTest, SubMatrix) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    Matrix m2(data2);
    Matrix m_sub = m1.sub(m2);
    std::vector<std::vector<float>> expected_data = {{-4.0f, -4.0f}, {-4.0f, -4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sub.equal(expected));
}

TEST(MatrixTest, MultiplyMatrix) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    Matrix m2(data2);
    Matrix m_multiply = m1.multiply(m2);
    std::vector<std::vector<float>> expected_data = {{5.0f, 12.0f}, {21.0f, 32.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_multiply.equal(expected));
}

TEST(MatrixTest, DivideMatrix) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    Matrix m2(data2);
    Matrix m_divide = m1.divide(m2);
    std::vector<std::vector<float>> expected_data = {{1.0f / 5.0f, 2.0f / 6.0f}, {3.0f / 7.0f, 4.0f / 8.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_divide.equal(expected));
}

TEST(MatrixTest, PowMatrix) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    Matrix m2 = Matrix::full(2, 2, 2.0f);
    Matrix m_pow = m1.pow(m2);
    std::vector<std::vector<float>> expected_data = {{1.0f, 4.0f}, {9.0f, 16.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_pow.equal(expected));
}

TEST(MatrixTest, AddScalar) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_add = m.add(1);
    std::vector<std::vector<float>> expected_data = {{2.0f, 3.0f}, {4.0f, 5.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_add.equal(expected));
}

TEST(MatrixTest, SubScalar) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_sub = m.sub(1);
    std::vector<std::vector<float>> expected_data = {{0.0f, 1.0f}, {2.0f, 3.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sub.equal(expected));
}

TEST(MatrixTest, MultiplyScalar) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_multiply = m.multiply(2);
    std::vector<std::vector<float>> expected_data = {{2.0f, 4.0f}, {6.0f, 8.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_multiply.equal(expected));
}

TEST(MatrixTest, DivideScalar) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_divide = m.divide(2);
    std::vector<std::vector<float>> expected_data = {{0.5f, 1.0f}, {1.5f, 2.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_divide.equal(expected));
}

TEST(MatrixTest, PowScalar) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_pow = m.pow(2);
    std::vector<std::vector<float>> expected_data = {{1.0f, 4.0f}, {9.0f, 16.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_pow.equal(expected));
}

TEST(MatrixTest, NegOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_neg = -m;
    std::vector<std::vector<float>> expected_data = {{-1.0f, -2.0f}, {-3.0f, -4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_neg.equal(expected));
}

TEST(MatrixTest, AddOperator) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    Matrix m2(data2);
    Matrix m_add = m1 + m2;
    std::vector<std::vector<float>> expected_data = {{6.0f, 8.0f}, {10.0f, 12.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_add.equal(expected));
}

TEST(MatrixTest, SubOperator) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    Matrix m2(data2);
    Matrix m_sub = m1 - m2;
    std::vector<std::vector<float>> expected_data = {{-4.0f, -4.0f}, {-4.0f, -4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sub.equal(expected));
}

TEST(MatrixTest, MultiplyOperator) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    Matrix m2(data2);
    Matrix m_multiply = m1 * m2;
    std::vector<std::vector<float>> expected_data = {{5.0f, 12.0f}, {21.0f, 32.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_multiply.equal(expected));
}

TEST(MatrixTest, DivideOperator) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7, 8}};
    Matrix m2(data2);
    Matrix m_divide = m1 / m2;
    std::vector<std::vector<float>> expected_data = {{1.0f / 5.0f, 2.0f / 6.0f}, {3.0f / 7.0f, 4.0f / 8.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_divide.equal(expected));
}

TEST(MatrixTest, AddScalarOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_add = m + 1.0f;
    std::vector<std::vector<float>> expected_data = {{2.0f, 3.0f}, {4.0f, 5.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_add.equal(expected));
}

TEST(MatrixTest, SubScalarOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_sub = m - 1.0f;
    std::vector<std::vector<float>> expected_data = {{0.0f, 1.0f}, {2.0f, 3.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sub.equal(expected));
}

TEST(MatrixTest, MultiplyScalarOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_multiply = m * 2.0f;
    std::vector<std::vector<float>> expected_data = {{2.0f, 4.0f}, {6.0f, 8.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_multiply.equal(expected));
}

TEST(MatrixTest, DivideScalarOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_divide = m / 2.0f;
    std::vector<std::vector<float>> expected_data = {{0.5f, 1}, {1.5f, 2.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_divide.equal(expected));
}

TEST(MatrixTest, PowScalarOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_pow = m.pow(2.0f);
    std::vector<std::vector<float>> expected_data = {{1.0f, 4.0f}, {9, 16.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_pow.equal(expected));
}

TEST(MatrixTest, LeftAddScalarOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_add = 1.0f + m;
    std::vector<std::vector<float>> expected_data = {{2, 3}, {4, 5}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_add.equal(expected));
}

TEST(MatrixTest, LeftSubScalarOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_sub = 1.0f - m;
    std::vector<std::vector<float>> expected_data = {{0, -1}, {-2, -3}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sub.equal(expected));
}

TEST(MatrixTest, LeftMultiplyScalarOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_multiply = 2.0f * m;
    std::vector<std::vector<float>> expected_data = {{2, 4.0f}, {6.0f, 8}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_multiply.equal(expected));
}

TEST(MatrixTest, LeftDivideScalarOperator) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_divide = 2.0f / m;
    std::vector<std::vector<float>> expected_data = {{2, 1}, {2.0f / 3.0f, 0.5}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_divide.equal(expected));
}

TEST(MatrixTest, Matmul) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7, 8}};
    Matrix m2(data2);
    Matrix m_matmul = m1.matmul(m2);
    std::vector<std::vector<float>> expected_data = {{19, 22.0f}, {43.0f, 50}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_matmul.equal(expected));
}

TEST(MatrixTest, Sqrt) {
    std::vector<std::vector<float>> data = {{1.0f, 4.0f}, {9, 16.0f}};
    Matrix m(data);
    Matrix m_sqrt = m.sqrt();
    std::vector<std::vector<float>> expected_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sqrt.equal(expected));
}

TEST(MatrixTest, Exp) {
    std::vector<std::vector<float>> data = {{0.0f, 1}, {2, 3}};
    Matrix m(data);
    Matrix m_exp = m.exp();
    std::vector<std::vector<float>> expected_data = {{1.0f, std::exp(1)}, {std::exp(2), std::exp(3)}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_exp.equal(expected));
}

TEST(MatrixTest, Log) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_log = m.log();
    std::vector<std::vector<float>> expected_data = {{0, std::log(2)}, {std::log(3), std::log(4)}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_log.equal(expected));
}

TEST(MatrixTest, Tanh) {
    std::vector<std::vector<float>> data = {{0, 1}, {2, 3}};
    Matrix m(data);
    Matrix m_tanh = m.tanh();
    std::vector<std::vector<float>> expected_data = {{0, std::tanh(1)}, {std::tanh(2), std::tanh(3)}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_tanh.equal(expected));
}

TEST(MatrixTest, Sigmoid) {
    std::vector<std::vector<float>> data = {{0, 1}, {2, 3}};
    Matrix m(data);
    Matrix m_sigmoid = m.sigmoid();
    std::vector<std::vector<float>> expected_data = {{0.5, 1 / (1 + std::exp(-1))}, {1 / (1 + std::exp(-2)), 1 / (1 + std::exp(-3))}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sigmoid.equal(expected));
}

TEST(MatrixTest, Sum) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_sum = m.sum();
    std::vector<std::vector<float>> expected_data = {{4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sum.equal(expected));
}

TEST(MatrixTest, SumAxis0) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_sum = m.sum(0);
    std::vector<std::vector<float>> expected_data = {{4.0f, 6.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sum.equal(expected));
}

TEST(MatrixTest, SumAxis1) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_sum = m.sum(1);
    std::vector<std::vector<float>> expected_data = {{3.0f}, {7.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_sum.equal(expected));
}

TEST(MatrixTest, Softmax) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_softmax = m.softmax();
    std::vector<std::vector<float>> expected_data = {{1 / (1 + std::exp(1)), 1 / (1 + std::exp(2))}, {1 / (1 + std::exp(3)), 1 / (1 + std::exp(4))}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_softmax.equal(expected));
}

TEST(MatrixTest, SoftmaxAxis0) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_softmax = m.softmax(0);
    std::vector<std::vector<float>> expected_data = {{1 / (1 + std::exp(1)), 1 / (1 + std::exp(3))}, {1 / (1 + std::exp(2)), 1 / (1 + std::exp(4))}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_softmax.equal(expected));
}

TEST(MatrixTest, SoftmaxAxis1) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix m_softmax = m.softmax(1);
    std::vector<std::vector<float>> expected_data = {{1 / (1 + std::exp(1)), 1 / (1 + std::exp(2))}, {1 / (1 + std::exp(3)), 1 / (1 + std::exp(4))}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m_softmax.equal(expected));
}

TEST(MatrixTest, Equal) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m2(data2);
    EXPECT_TRUE(m1.equal(m2));
}

TEST(MatrixTest, NotEqual) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{1.0f, 2.0f}, {3.0f, 5}};
    Matrix m2(data2);
    EXPECT_FALSE(m1.equal(m2));
}

TEST(MatrixTest, AllClose) {
    float tolerance = 1e-5;
    float within_tolerance = 1e-6.0f;
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m2(data2);
    m2 = m2 + within_tolerance;
    EXPECT_TRUE(m1.all_close(m2, tolerance));
}

TEST(MatrixTest, NotAllClose) {
    float tolerance = 1e-5;
    float beyond_tolerance = 1e-4;
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m2(data2);
    m2 = m2 + beyond_tolerance;
    EXPECT_FALSE(m1.all_close(m2, tolerance));
}

TEST(MatrixTest, Set) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    m.set(0, 0, 5);
    std::vector<std::vector<float>> expected_data = {{5.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix expected(expected_data);
    EXPECT_TRUE(m.equal(expected));
}

TEST(MatrixTest, At) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    EXPECT_EQ(m.at(0, 0), 1);
    EXPECT_EQ(m.at(0, 1), 2);
    EXPECT_EQ(m.at(1.0f, 0), 3);
    EXPECT_EQ(m.at(1.0f, 1), 4);
}

TEST(MatrixTest, Scalar) {
    std::vector<std::vector<float>> data = {{1.0f}};
    Matrix m(data);
    EXPECT_EQ(m.scalar(), 1.0f);
}

TEST(MatrixTest, DimsSameAs) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    Matrix m2(data2);
    EXPECT_TRUE(m1.dims_same_as(m2));
}

TEST(MatrixTest, DimsNotSameAs) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}};
    Matrix m2(data2);
    EXPECT_FALSE(m1.dims_same_as(m2));
}

TEST(MatrixTest, InnerDimSameAs) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    Matrix m2(data2);
    EXPECT_TRUE(m1.inner_dim_same_as(m2));
}

TEST(MatrixTest, InnerDimNotSameAs) {
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m1(data1);
    std::vector<std::vector<float>> data2 = {{5.0f, 6.0f}};
    Matrix m2(data2);
    EXPECT_FALSE(m1.inner_dim_same_as(m2));
}
