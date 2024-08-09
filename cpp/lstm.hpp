#pragma once

#include <unordered_map>

#include "matrix.hpp"

class LSTM {
public:
    static LSTM load(const std::string &model_path);

public:
    LSTM() = delete;
    LSTM(const LSTM &);
    LSTM &operator=(const LSTM &) = delete;
    LSTM &operator=(LSTM &&) = delete;

    LSTM(uint32_t input_size, uint32_t hidden_size, uint32_t output_size, float learning_rate = 1e-3);

    std::vector<Matrix> forward(const std::vector<Matrix> &inputs);
    void backward(const std::vector<Matrix> &labels);

    void save(const std::string &model_path) const;

private:
    void reset_cache();

private:
    // Hyperparameters
    uint32_t input_size;
    uint32_t hidden_size;
    uint32_t output_size;

    float learning_rate;

    // Forget gate weights
    Matrix W_f;
    Matrix b_f;

    // Input gate weights
    Matrix W_i;
    Matrix b_i;

    // Candidate gate weights
    Matrix W_c;
    Matrix b_c;

    // Output gate weights
    Matrix W_o;
    Matrix b_o;

    // Final gate weights
    Matrix W_y;
    Matrix b_y;

    // Network cache
    std::unordered_map<int32_t, Matrix> concat_inputs;
    std::unordered_map<int32_t, Matrix> hidden_states;
    std::unordered_map<int32_t, Matrix> cell_states;
    std::unordered_map<int32_t, Matrix> forget_gates;
    std::unordered_map<int32_t, Matrix> input_gates;
    std::unordered_map<int32_t, Matrix> candidate_gates;
    std::unordered_map<int32_t, Matrix> output_gates;
    std::unordered_map<int32_t, Matrix> activation_outputs;
    std::unordered_map<int32_t, Matrix> outputs;
};