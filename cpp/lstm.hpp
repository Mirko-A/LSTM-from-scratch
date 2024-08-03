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
};