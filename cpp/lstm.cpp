#include <cmath>

#include "lstm.hpp"

LSTM::LSTM(uint32_t input_size, uint32_t hidden_size, uint32_t output_size, float learning_rate = 1e-3)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate) {
    W_f = std::move(init_weights(input_size, hidden_size));
    b_f = Matrix::zeros(hidden_size, 1);

    W_i = std::move(init_weights(input_size, hidden_size));
    b_i = Matrix::zeros(hidden_size, 1);

    W_c = std::move(init_weights(input_size, hidden_size));
    b_c = Matrix::zeros(hidden_size, 1);

    W_o = std::move(init_weights(input_size, hidden_size));
    b_o = Matrix::zeros(hidden_size, 1);

    W_y = std::move(init_weights(hidden_size, output_size));
    b_y = Matrix::zeros(output_size, 1);

    reset_cache();
}

void LSTM::reset_cache() {
    concat_inputs.clear();

    hidden_states.clear();
    hidden_states[-1] = Matrix::zeros(hidden_size, 1);
    cell_states.clear();
    cell_states[-1] = Matrix::zeros(hidden_size, 1);

    forget_gates.clear();
    input_gates.clear();
    candidate_gates.clear();
    output_gates.clear();
    activation_outputs.clear();
    outputs.clear();
}

std::vector<Matrix> LSTM::forward(std::vector<Matrix> inputs) {
    reset_cache();

    std::vector<Matrix> outputs;
    outputs.reserve(inputs.size());

    for (uint32_t i = 0; i < inputs.size(); ++i) {
        // todo
    }
}

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
    // Xavier initialization
    return Matrix::uniform(output_size, input_size, -1.0f, 1.0f) * std::sqrt(6.0f / (input_size + output_size));
}
