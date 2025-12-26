#pragma once

#include "matrix.hpp"

class LSTM {
  public:
    static LSTM load(const std::string &model_path);

  public:
    LSTM() = delete;
    LSTM(const LSTM &);
    LSTM &operator=(const LSTM &) = delete;
    LSTM &operator=(LSTM &&) = delete;

    LSTM(uint32_t input_size, uint32_t hidden_size, uint32_t output_size,
         float learning_rate = 1e-3);

    std::vector<Matrix> forward(const std::vector<Matrix> &inputs);
    void backward(const std::vector<Matrix> &labels);

    void save(const std::string &model_path) const;

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
    std::vector<Matrix> concat_inputs;
    std::vector<Matrix> hidden_states;
    std::vector<Matrix> cell_states;
    std::vector<Matrix> forget_gates;
    std::vector<Matrix> input_gates;
    std::vector<Matrix> candidate_gates;
    std::vector<Matrix> output_gates;
    std::vector<Matrix> activation_outputs;
    std::vector<Matrix> outputs;
};
