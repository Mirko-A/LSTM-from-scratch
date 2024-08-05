#include <cassert>
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

static Matrix init_weights(uint32_t input_size, uint32_t output_size) {
    // Xavier initialization
    return Matrix::uniform(output_size, input_size, -1.0f, 1.0f) * std::sqrt(6.0f / (input_size + output_size));
}

LSTM::LSTM(uint32_t input_size, uint32_t hidden_size, uint32_t output_size, float learning_rate)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate) {
    W_f = init_weights(input_size, hidden_size);
    b_f = Matrix::zeros(hidden_size, 1);

    W_i = init_weights(input_size, hidden_size);
    b_i = Matrix::zeros(hidden_size, 1);

    W_c = init_weights(input_size, hidden_size);
    b_c = Matrix::zeros(hidden_size, 1);

    W_o = init_weights(input_size, hidden_size);
    b_o = Matrix::zeros(hidden_size, 1);

    W_y = init_weights(hidden_size, output_size);
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

    std::vector<Matrix> ret_outputs(inputs.size());

    for (uint32_t t = 0; t < inputs.size(); ++t) {
        concat_inputs[t] = Matrix::concatenate(0, {hidden_states[t - 1], inputs[t]});

        forget_gates[t] = W_f.matmul(concat_inputs[t]) + b_f;
        input_gates[t] = W_i.matmul(concat_inputs[t]) + b_i;
        candidate_gates[t] = W_c.matmul(concat_inputs[t]) + b_c;
        output_gates[t] = W_o.matmul(concat_inputs[t]) + b_o;

        Matrix fga = forget_gates[t].sigmoid();
        Matrix iga = input_gates[t].sigmoid();
        Matrix cga = candidate_gates[t].tanh();
        Matrix oga = output_gates[t].sigmoid();

        cell_states[t] = fga * cell_states[t - 1] + iga * cga;
        hidden_states[t] = oga * cell_states[t].tanh();

        Matrix output = (W_y.matmul(hidden_states[t]) + b_y).softmax(1);
        ret_outputs.push_back(output);
        outputs[t] = output;
    }

    return ret_outputs;
}

void LSTM::backward(std::vector<Matrix> labels) {
    Matrix dW_f = Matrix::zeros_like(W_f);
    Matrix db_f = Matrix::zeros_like(b_f);
    Matrix dW_i = Matrix::zeros_like(W_i);
    Matrix db_i = Matrix::zeros_like(b_i);
    Matrix dW_c = Matrix::zeros_like(W_c);
    Matrix db_c = Matrix::zeros_like(b_c);
    Matrix dW_o = Matrix::zeros_like(W_o);
    Matrix db_o = Matrix::zeros_like(b_o);
    Matrix dW_y = Matrix::zeros_like(W_y);
    Matrix db_y = Matrix::zeros_like(b_y);

    Matrix hidden_state = Matrix::zeros_like(hidden_states[0]);
    Matrix cell_state = Matrix::zeros_like(cell_states[0]);

    for (int32_t t = concat_inputs.size() - 1; t >= 0; t--) {
        Matrix dL = -(labels[t] / outputs[t]);
        Matrix dsoftmax = outputs[t] * dL - outputs[t].T().matmul(dL) * outputs[t];

        dW_y = dW_y + dsoftmax.matmul(hidden_states[t].T());
        db_y = db_y + dsoftmax;

        Matrix dhidden_state = W_y.T().matmul(dsoftmax) + hidden_state;
        Matrix doutput = cell_states[t].tanh() * dhidden_state * dsigmoid(output_gates[t]);

        dW_o = dW_o + doutput.matmul(concat_inputs[t].T());
        db_o = db_o + doutput;

        Matrix dcell_state = dtanh(cell_states[t]) * output_gates[t].sigmoid() + dhidden_state + cell_state;

        Matrix dforget = dcell_state * cell_states[t - 1] * dsigmoid(forget_gates[t]);

        dW_f = dW_f + dforget.matmul(concat_inputs[t].T());
        db_f = db_f + dforget;

        Matrix dinput = dcell_state * candidate_gates[t].tanh() * dsigmoid(input_gates[t]);

        dW_i = dW_i + dinput.matmul(concat_inputs[t].T());
        db_i = db_i + dinput;

        Matrix dcandidate = dcell_state * input_gates[t].sigmoid() + dtanh(candidate_gates[t]);

        dW_c = dW_c + dcandidate.matmul(concat_inputs[t].T());
        db_c = db_c + dcandidate;

        Matrix dconcat_inputs = W_f.T().matmul(dforget) +
                                W_i.T().matmul(dinput) +
                                W_c.T().matmul(dcandidate) +
                                W_o.T().matmul(doutput);

        uint32_t shrink_size = dconcat_inputs.shape().first - hidden_size;
        hidden_state = dconcat_inputs.shrink_end(0, shrink_size);
        cell_state = forget_gates[t].sigmoid() * dcell_state;
    }

    dW_f = dW_f.clamp(-1.0f, 1.0f);
    db_f = db_f.clamp(-1.0f, 1.0f);
    dW_i = dW_i.clamp(-1.0f, 1.0f);
    db_i = db_i.clamp(-1.0f, 1.0f);
    dW_c = dW_c.clamp(-1.0f, 1.0f);
    db_c = db_c.clamp(-1.0f, 1.0f);
    dW_o = dW_o.clamp(-1.0f, 1.0f);
    db_o = db_o.clamp(-1.0f, 1.0f);
    dW_y = dW_y.clamp(-1.0f, 1.0f);
    db_y = db_y.clamp(-1.0f, 1.0f);

    W_f = W_f - dW_f * learning_rate;
    b_f = b_f - db_f * learning_rate;
    W_i = W_i - dW_i * learning_rate;
    b_i = b_i - db_i * learning_rate;
    W_c = W_c - dW_c * learning_rate;
    b_c = b_c - db_c * learning_rate;
    W_o = W_o - dW_o * learning_rate;
    b_o = b_o - db_o * learning_rate;
    W_y = W_y - dW_y * learning_rate;
    b_y = b_y - db_y * learning_rate;
}

std::vector<Matrix> LSTM::train(const std::vector<Matrix> &one_hot_inputs, const std::vector<Matrix> &one_hot_labels, uint32_t vocab_size, uint32_t epochs) {
    assert(one_hot_inputs.size() == one_hot_labels.size());
    uint32_t data_size = static_cast<uint32_t>(one_hot_inputs.size());

    std::vector<Matrix> losses(data_size);
    for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
        std::vector<Matrix> predictions = forward(one_hot_inputs);
        uint32_t N = static_cast<uint32_t>(predictions.size());

        Matrix loss = Matrix::zeros(1, 1);

        for (uint32_t i = 0; i < N; ++i) {
            loss = loss + cross_entropy_loss(predictions[i], one_hot_labels[i]);
        }

        losses[epoch] = loss / N;
        backward(one_hot_labels);
    }

    return losses;
}

std::tuple<std::string, float> test(const std::vector<Matrix> &one_hot_inputs, const std::vector<Matrix> &one_hot_labels) {
    return std::make_tuple(std::string(), 0.0f);
}
