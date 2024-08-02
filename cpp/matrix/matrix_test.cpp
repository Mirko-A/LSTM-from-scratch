#include "matrix.hpp"
#include <gtest/gtest.h>

TEST(MatrixTest, Constructor) {
    std::vector<std::vector<float>> data = {{1, 2}, {3, 4}};
    Matrix m(data);
    EXPECT_EQ(m.at(0, 0), 1);
    EXPECT_EQ(m.at(0, 1), 2);
    EXPECT_EQ(m.at(1, 0), 3);
    EXPECT_EQ(m.at(1, 1), 4);
}

TEST(MatrixTest, Zeros) {
    Matrix m = Matrix::zeros(2, 2);
    EXPECT_EQ(m.at(0, 0), 0);
    EXPECT_EQ(m.at(0, 1), 0);
    EXPECT_EQ(m.at(1, 0), 0);
    EXPECT_EQ(m.at(1, 1), 0);
}

TEST(MatrixTest, Ones) {
    Matrix m = Matrix::ones(2, 2);
    EXPECT_EQ(m.at(0, 0), 1);
    EXPECT_EQ(m.at(0, 1), 1);
    EXPECT_EQ(m.at(1, 0), 1);
    EXPECT_EQ(m.at(1, 1), 1);
}

TEST(MatrixTest, Full) {
    Matrix m = Matrix::full(2, 2, 3.5);
    EXPECT_EQ(m.at(0, 0), 3.5);
    EXPECT_EQ(m.at(0, 1), 3.5);
    EXPECT_EQ(m.at(1, 0), 3.5);
    EXPECT_EQ(m.at(1, 1), 3.5);
}

TEST(MatrixTest, FullLike) {
    std::vector<std::vector<float>> data = {{1, 2}, {3, 4}};
    Matrix m(data);
    Matrix m_full_like = Matrix::full_like(m, 5.0);
    EXPECT_EQ(m_full_like.at(0, 0), 5.0);
    EXPECT_EQ(m_full_like.at(0, 1), 5.0);
    EXPECT_EQ(m_full_like.at(1, 0), 5.0);
    EXPECT_EQ(m_full_like.at(1, 1), 5.0);
}

TEST(MatrixTest, ZerosLike) {
    std::vector<std::vector<float>> data = {{1, 2}, {3, 4}};
    Matrix m(data);
    Matrix m_zeros_like = Matrix::zeros_like(m);
    EXPECT_EQ(m_zeros_like.at(0, 0), 0.0);
    EXPECT_EQ(m_zeros_like.at(0, 1), 0.0);
    EXPECT_EQ(m_zeros_like.at(1, 0), 0.0);
    EXPECT_EQ(m_zeros_like.at(1, 1), 0.0);
}

TEST(MatrixTest, OnesLike) {
    std::vector<std::vector<float>> data = {{1, 2}, {3, 4}};
    Matrix m(data);
    Matrix m_ones_like = Matrix::ones_like(m);
    EXPECT_EQ(m_ones_like.at(0, 0), 1.0);
    EXPECT_EQ(m_ones_like.at(0, 1), 1.0);
    EXPECT_EQ(m_ones_like.at(1, 0), 1.0);
    EXPECT_EQ(m_ones_like.at(1, 1), 1.0);
}

TEST(MatrixTest, Uniform) {
    Matrix m = Matrix::uniform(2, 2, 0.0, 1.0);
    EXPECT_GE(m.at(0, 0), 0.0);
    EXPECT_LE(m.at(0, 0), 1.0);
    EXPECT_GE(m.at(0, 1), 0.0);
    EXPECT_LE(m.at(0, 1), 1.0);
    EXPECT_GE(m.at(1, 0), 0.0);
    EXPECT_LE(m.at(1, 0), 1.0);
    EXPECT_GE(m.at(1, 1), 0.0);
    EXPECT_LE(m.at(1, 1), 1.0);
}

TEST(MatrixTest, Transpose) {
    std::vector<std::vector<float>> data = {{1, 2}, {3, 4}};
    Matrix m(data);
    Matrix t = m.transpose();
    EXPECT_EQ(t.at(0, 0), 1);
    EXPECT_EQ(t.at(0, 1), 3);
    EXPECT_EQ(t.at(1, 0), 2);
    EXPECT_EQ(t.at(1, 1), 4);
}

TEST(MatrixTest, Expand) {
    std::vector<std::vector<float>> data = {{1, 2}};
    Matrix m(data);
    Matrix expanded = m.expand(0, 2);
    EXPECT_EQ(expanded.at(0, 0), 1);
    EXPECT_EQ(expanded.at(0, 1), 2);
    EXPECT_EQ(expanded.at(1, 0), 1);
    EXPECT_EQ(expanded.at(1, 1), 2);
}

TEST(MatrixTest, Neg) {
    std::vector<std::vector<float>> data = {{1, -2}, {-3, 4}};
    Matrix m(data);
    Matrix neg_m = m.neg();
    EXPECT_EQ(neg_m.at(0, 0), -1);
    EXPECT_EQ(neg_m.at(0, 1), 2);
    EXPECT_EQ(neg_m.at(1, 0), 3);
    EXPECT_EQ(neg_m.at(1, 1), -4);
}

TEST(MatrixTest, Add) {
    std::vector<std::vector<float>> data1 = {{1, 2}, {3, 4}};
    std::vector<std::vector<float>> data2 = {{5, 6}, {7, 8}};
    Matrix m1(data1);
    Matrix m2(data2);
    Matrix result = m1.add(m2);
    EXPECT_EQ(result.at(0, 0), 6);
    EXPECT_EQ(result.at(0, 1), 8);
    EXPECT_EQ(result.at(1, 0), 10);
    EXPECT_EQ(result.at(1, 1), 12);
}

TEST(MatrixTest, Sub) {
    std::vector<std::vector<float>> data1 = {{5, 6}, {7, 8}};
    std::vector<std::vector<float>> data2 = {{1, 2}, {3, 4}};
    Matrix m1(data1);
    Matrix m2(data2);
    Matrix result = m1.sub(m2);
    EXPECT_EQ(result.at(0, 0), 4);
    EXPECT_EQ(result.at(0, 1), 4);
    EXPECT_EQ(result.at(1, 0), 4);
    EXPECT_EQ(result.at(1, 1), 4);
}

TEST(MatrixTest, Multiply) {
    std::vector<std::vector<float>> data1 = {{1, 2}, {3, 4}};
    std::vector<std::vector<float>> data2 = {{5, 6}, {7, 8}};
    Matrix m1(data1);
    Matrix m2(data2);
    Matrix result = m1.multiply(m2);
    EXPECT_EQ(result.at(0, 0), 5);
    EXPECT_EQ(result.at(0, 1), 12);
    EXPECT_EQ(result.at(1, 0), 21);
    EXPECT_EQ(result.at(1, 1), 32);
}

TEST(MatrixTest, Divide) {
    std::vector<std::vector<float>> data1 = {{10, 20}, {30, 40}};
    std::vector<std::vector<float>> data2 = {{2, 4}, {5, 8}};
    Matrix m1(data1);
    Matrix m2(data2);
    Matrix result = m1.divide(m2);
    EXPECT_EQ(result.at(0, 0), 5);
    EXPECT_EQ(result.at(0, 1), 5);
    EXPECT_EQ(result.at(1, 0), 6);
    EXPECT_EQ(result.at(1, 1), 5);
}

TEST(MatrixTest, Matmul) {
    std::vector<std::vector<float>> data1 = {{1, 2}, {3, 4}};
    std::vector<std::vector<float>> data2 = {{5, 6}, {7, 8}};
    Matrix m1(data1);
    Matrix m2(data2);
    Matrix result = m1.matmul(m2);
    EXPECT_EQ(result.at(0, 0), 19);
    EXPECT_EQ(result.at(0, 1), 22);
    EXPECT_EQ(result.at(1, 0), 43);
    EXPECT_EQ(result.at(1, 1), 50);
}

TEST(MatrixTest, Pow) {
    std::vector<std::vector<float>> data1 = {{1, 2}, {3, 4}};
    std::vector<std::vector<float>> data2 = {{2, 2}, {2, 2}};
    Matrix m1(data1);
    Matrix m2(data2);
    Matrix result = m1.pow(m2);
    EXPECT_EQ(result.at(0, 0), 1);
    EXPECT_EQ(result.at(0, 1), 4);
    EXPECT_EQ(result.at(1, 0), 9);
    EXPECT_EQ(result.at(1, 1), 16);
}

TEST(MatrixTest, Sqrt) {
    std::vector<std::vector<float>> data = {{1, 4}, {9, 16}};
    Matrix m(data);
    Matrix result = m.sqrt();
    EXPECT_EQ(result.at(0, 0), 1);
    EXPECT_EQ(result.at(0, 1), 2);
    EXPECT_EQ(result.at(1, 0), 3);
    EXPECT_EQ(result.at(1, 1), 4);
}

TEST(MatrixTest, Exp) {
    std::vector<std::vector<float>> data = {{0, 1}, {2, 3}};
    Matrix m(data);
    Matrix result = m.exp();
    EXPECT_NEAR(result.at(0, 0), 1, 1e-5);
    EXPECT_NEAR(result.at(0, 1), 2.71828, 1e-5);
    EXPECT_NEAR(result.at(1, 0), 7.38906, 1e-5);
    EXPECT_NEAR(result.at(1, 1), 20.0855, 5e-5);
}

TEST(MatrixTest, Log) {
    std::vector<std::vector<float>> data = {{1, 2.71828}, {7.38906, 20.0855}};
    Matrix m(data);
    Matrix result = m.log();
    EXPECT_NEAR(result.at(0, 0), 0, 1e-5);
    EXPECT_NEAR(result.at(0, 1), 1, 1e-5);
    EXPECT_NEAR(result.at(1, 0), 2, 1e-5);
    EXPECT_NEAR(result.at(1, 1), 3, 1e-5);
}

TEST(MatrixTest, Tanh) {
    std::vector<std::vector<float>> data = {{0, 1}, {-1, 2}};
    Matrix m(data);
    Matrix result = m.tanh();
    EXPECT_NEAR(result.at(0, 0), 0, 1e-5);
    EXPECT_NEAR(result.at(0, 1), 0.761594, 1e-5);
    EXPECT_NEAR(result.at(1, 0), -0.761594, 1e-5);
    EXPECT_NEAR(result.at(1, 1), 0.964028, 1e-5);
}

TEST(MatrixTest, Sigmoid) {
    std::vector<std::vector<float>> data = {{0, 1}, {-1, 2}};
    Matrix m(data);
    Matrix result = m.sigmoid();
    EXPECT_NEAR(result.at(0, 0), 0.5, 1e-5);
    EXPECT_NEAR(result.at(0, 1), 0.731059, 1e-5);
    EXPECT_NEAR(result.at(1, 0), 0.268941, 1e-5);
    EXPECT_NEAR(result.at(1, 1), 0.880797, 1e-5);
}

TEST(MatrixTest, Sum) {
    std::vector<std::vector<float>> data = {{1, 2}, {3, 4}};
    Matrix m(data);
    Matrix result = m.sum(std::nullopt);
    EXPECT_EQ(result.at(0, 0), 10);

    Matrix result_axis0 = m.sum(0);
    EXPECT_EQ(result_axis0.at(0, 0), 4);
    EXPECT_EQ(result_axis0.at(0, 1), 6);

    Matrix result_axis1 = m.sum(1);
    EXPECT_EQ(result_axis1.at(0, 0), 3);
    EXPECT_EQ(result_axis1.at(1, 0), 7);
}

TEST(MatrixTest, Softmax) {
    std::vector<std::vector<float>> data = {{1, 2}, {3, 4}};
    Matrix m(data);
    Matrix result = m.softmax(std::nullopt);
    float sum = result.at(0, 0) + result.at(0, 1) + result.at(1, 0) + result.at(1, 1);
    EXPECT_NEAR(sum, 1.0, 1e-5);

    Matrix result_axis0 = m.softmax(0);
    EXPECT_NEAR(result_axis0.at(0, 0) + result_axis0.at(1, 0), 1.0, 1e-5);
    EXPECT_NEAR(result_axis0.at(0, 1) + result_axis0.at(1, 1), 1.0, 1e-5);

    Matrix result_axis1 = m.softmax(1);
    EXPECT_NEAR(result_axis1.at(0, 0) + result_axis1.at(0, 1), 1.0, 1e-5);
    EXPECT_NEAR(result_axis1.at(1, 0) + result_axis1.at(1, 1), 1.0, 1e-5);
}

TEST(MatrixTest, Equal) {
    std::vector<std::vector<float>> data1 = {{1, 2}, {3, 4}};
    std::vector<std::vector<float>> data2 = {{1, 2}, {3, 4}};
    Matrix m1(data1);
    Matrix m2(data2);
    EXPECT_TRUE(m1.equal(m2));
}

TEST(MatrixTest, AllClose) {
    std::vector<std::vector<float>> data1 = {{1.00001, 2.00001}, {3.00001, 4.00001}};
    std::vector<std::vector<float>> data2 = {{1.00002, 2.00002}, {3.00002, 4.00002}};
    Matrix m1(data1);
    Matrix m2(data2);
    EXPECT_TRUE(m1.all_close(m2, 1e-4));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}