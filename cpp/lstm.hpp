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
    uint32_t input_size;
    uint32_t hidden_size;
    uint32_t output_size;

    float learning_rate;

    Matrix W_f;
    Matrix b_f;

    Matrix W_i;
    Matrix b_i;

    Matrix W_c;
    Matrix b_c;

    Matrix W_o;
    Matrix b_o;

    Matrix W_y;
    Matrix b_y;

    std::unordered_map<int, Matrix> cell_states;
    std::unordered_map<int, Matrix> forget_gates;
    std::unordered_map<int, Matrix> input_gates;
    std::unordered_map<int, Matrix> candidate_gates;
    std::unordered_map<int, Matrix> output_gates;
    std::unordered_map<int, Matrix> activation_outputs;
    std::unordered_map<int, Matrix> outputs;
};