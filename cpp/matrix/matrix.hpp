#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

class Matrix {
public:
    Matrix() = delete;
    Matrix(std::vector<std::vector<float>> data);

    static Matrix full(uint32_t row_n, uint32_t col_n, float value);
    static Matrix zeros(uint32_t row_n, uint32_t col_n);
    static Matrix ones(uint32_t row_n, uint32_t col_n);
    static Matrix full_like(const Matrix &other, float value);
    static Matrix zeros_like(const Matrix &other);
    static Matrix ones_like(const Matrix &other);
    static Matrix arange(uint32_t row_n, uint32_t col_n, uint32_t start = 0);
    static Matrix uniform(uint32_t row_n, uint32_t col_n, float low, float high);

    Matrix transpose() const;
    Matrix T() const;
    Matrix expand(int axis, uint32_t new_size) const;

    Matrix neg() const;
    Matrix add(const Matrix &other) const;
    Matrix sub(const Matrix &other) const;
    Matrix multiply(const Matrix &other) const;
    Matrix divide(const Matrix &other) const;
    Matrix pow(const Matrix &other) const;

    Matrix add(float scalar) const;
    Matrix sub(float scalar) const;
    Matrix multiply(float scalar) const;
    Matrix divide(float scalar) const;
    Matrix pow(float scalar) const;

    Matrix operator-() const;

    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;
    Matrix operator/(const Matrix &other) const;

    friend Matrix operator+(float scalar, const Matrix &matrix);
    friend Matrix operator-(float scalar, const Matrix &matrix);
    friend Matrix operator*(float scalar, const Matrix &matrix);
    friend Matrix operator/(float scalar, const Matrix &matrix);

    Matrix operator+(float scalar) const;
    Matrix operator-(float scalar) const;
    Matrix operator*(float scalar) const;
    Matrix operator/(float scalar) const;

    Matrix matmul(const Matrix &other) const;

    Matrix sqrt() const;
    Matrix exp() const;
    Matrix log() const;
    Matrix tanh() const;
    Matrix sigmoid() const;

    Matrix sum(std::optional<uint8_t> axis = std::nullopt) const;
    Matrix softmax(std::optional<uint8_t> axis = std::nullopt) const;

    bool equal(const Matrix &other) const;
    bool all_close(const Matrix &other, float tolerance = 1e-5) const;

    void set(uint32_t row_i, uint32_t col_i, float value);
    float at(uint32_t row_i, uint32_t col_i) const;
    float scalar() const;

    bool dims_same_as(const Matrix &other) const;
    bool inner_dim_same_as(const Matrix &other) const;

    void print() const;
    void println() const;
    friend std::ostream &operator<<(std::ostream &os, const Matrix &rhs);

private:
    uint32_t row_n;
    uint32_t col_n;
    std::vector<std::vector<float>> data;
};