#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>

#include "lstm.hpp"
#include "matrix.hpp"

static std::optional<std::string> read_to_string(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return std::nullopt;
    }

    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content.append(line);
        content.push_back('\n');
    }

    return std::make_optional(content);
}

static std::set<char> create_vocab(const std::vector<char> &data) {
    return std::set<char>(data.begin(), data.end());
}

static Matrix one_hot_encode(uint32_t class_idx, uint32_t n_classes) {
    assert(class_idx < n_classes);

    Matrix one_hot = Matrix::zeros(n_classes, 1);
    one_hot.set(class_idx, 0, 1.0f);

    return one_hot;
}

int main() {
    std::string dataset_path = "../../datasets/shakespeare/tiny_shakespeare_small.txt";
    std::optional<std::string> maybe_dataset = read_to_string(dataset_path);

    if (!maybe_dataset.has_value()) {
        std::cerr << "Failed to read file: " << dataset_path << std::endl;
        return 1;
    }

    std::string dataset_str = maybe_dataset.value();
    std::vector<char> dataset(dataset_str.begin(), dataset_str.end());
    uint32_t dataset_size = static_cast<uint32_t>(dataset.size());
    std::set<char> vocab = create_vocab(dataset);
    uint32_t vocab_size = static_cast<uint32_t>(vocab.size());

    std::cout << "Data size: " << dataset_size << std::endl;
    std::cout << "Vocab size: " << vocab_size << std::endl;

    std::unordered_map<char, uint32_t> char_to_idx;
    std::unordered_map<uint32_t, char> idx_to_char;

    uint32_t idx = 0;
    for (char c : vocab) {
        char_to_idx[c] = idx;
        idx_to_char[idx] = c;
        idx += 1;
    }

    std::vector<char> x_train_chars(dataset.begin(), dataset.end() - 1);
    std::vector<char> y_train_chars(dataset.begin() + 1, dataset.end());

    std::vector<Matrix> x_train(x_train_chars.size());
    std::vector<Matrix> y_train(y_train_chars.size());

    assert(x_train.size() == y_train.size());

    uint32_t data_size = static_cast<uint32_t>(x_train.size());

    for (uint32_t i = 0; i < data_size; ++i) {
        x_train[i] = one_hot_encode(char_to_idx[x_train_chars[i]], vocab_size);
        y_train[i] = one_hot_encode(char_to_idx[y_train_chars[i]], vocab_size);
    }

    // Hyperparameters
    uint32_t hidden_size = 64;
    uint32_t input_size = vocab_size + hidden_size;
    uint32_t output_size = vocab_size;

    float learning_rate = 0.06f;
    uint32_t epochs = 10;

    LSTM lstm(input_size, hidden_size, output_size, learning_rate);

    std::cout << "Training LSTM network..." << std::endl;

    auto losses = lstm.train(x_train, y_train, vocab_size, epochs);

    uint32_t epoch = 0;
    for (auto loss : losses) {
        std::cout << "Loss epoch " << epoch << ": " << loss << std::endl;
        epoch++;
    }

    return 0;
}