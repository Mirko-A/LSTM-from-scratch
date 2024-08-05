#include <unordered_map>

#include "matrix.hpp"

class LSTM {
public:
    LSTM(uint32_t input_size, uint32_t hidden_size, uint32_t output_size, float learning_rate = 1e-3);
    std::vector<Matrix> forward(std::vector<Matrix> inputs);
    void backward(std::vector<Matrix> labels);
    std::vector<Matrix> train(std::string inputs, uint32_t epochs = 1);
    std::tuple<std::string, float> test(std::string inputs, std::string labels);

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