#include <cmath>

#include "lstm.hpp"

static Matrix dtanh(const Matrix &x) {
    return 1.0f - x.tanh().pow(2.0f);
}

static Matrix dsigmoid(const Matrix &x) {
    return x.sigmoid() * (1.0f - x.sigmoid());
}

static Matrix cross_entropy_loss(const Matrix &y_pred, const Matrix &y_true) {
    return -(y_true * y_pred.log()).sum();
}

static Matrix one_hot_encode(uint32_t class_idx, uint32_t n_classes) {
    Matrix one_hot = Matrix::zeros(1, n_classes);
    one_hot.set(0, class_idx, 1.0f);

    return one_hot;
}

static Matrix init_weights(uint32_t input_size, uint32_t output_size) {
    return Matrix::uniform(output_size, input_size, -1.0f, 1.0f) * std::sqrt(6.0f / (input_size + output_size));
}
