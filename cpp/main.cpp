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

static Matrix cross_entropy_loss(const Matrix &y_pred, const Matrix &y_true) {
    return -(y_true * y_pred.log()).sum();
}

static std::vector<float> train(LSTM &model, const std::vector<Matrix> &one_hot_inputs, const std::vector<Matrix> &one_hot_labels, uint32_t vocab_size, uint32_t epochs) {
    assert(one_hot_inputs.size() == one_hot_labels.size());
    uint32_t data_size = static_cast<uint32_t>(one_hot_inputs.size());

    std::vector<float> losses;
    losses.reserve(data_size);
    for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
        std::vector<Matrix> predictions = model.forward(one_hot_inputs);
        uint32_t N = static_cast<uint32_t>(predictions.size());

        float loss = 0.0f;

        for (uint32_t i = 0; i < N; ++i) {
            loss = loss + cross_entropy_loss(predictions[i], one_hot_labels[i]).scalar();
        }

        float avg_loss = loss / N;
        std::cout << "Epoch " << epoch + 1 << " - loss: " << avg_loss << std::endl;

        losses.push_back(avg_loss);
        model.backward(one_hot_labels);
    }

    return losses;
}

uint32_t random_choice(const std::vector<float> &probabilities) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float cumulative_prob = 0.0f;

    for (size_t i = 0; i < probabilities.size(); i++) {
        cumulative_prob += probabilities[i];
        if (r <= cumulative_prob) {
            return static_cast<uint32_t>(i);
        }
    }

    return static_cast<uint32_t>(probabilities.size() - 1);
}

static std::tuple<std::string, float> test(LSTM &model,
                                           const std::vector<Matrix> &one_hot_inputs,
                                           const std::vector<char> &labels,
                                           std::unordered_map<uint32_t, char> &idx_to_char) {
    assert(one_hot_inputs.size() == labels.size());
    uint32_t accuracy = 0;
    auto probabilities = model.forward(one_hot_inputs);
    assert(probabilities.size() == labels.size());

    std::string output = "";

    for (size_t i = 0; i < one_hot_inputs.size(); i++) {
        Matrix prob_mat = probabilities[i];
        std::vector<float> probs = prob_mat.flatten_row().get_data()[0];

        char prediction = idx_to_char[random_choice(probs)];
        output.push_back(prediction);

        if (prediction == labels[i]) {
            accuracy++;
        }
    }

    float accuracy_perc = accuracy * 100 / one_hot_inputs.size();

    return std::make_tuple(output, accuracy_perc);
}

int main() {
    std::string dataset_path = "../../datasets/shakespeare/tiny_shakespeare_tiny.txt";
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

    float learning_rate = 0.005f;
    uint32_t epochs = 3;

    LSTM lstm(input_size, hidden_size, output_size, learning_rate);

    std::cout << "Training LSTM network..." << std::endl;

    auto losses = train(lstm, x_train, y_train, vocab_size, epochs);

    const std::string model_path = "./model/lstm.json";
    lstm.save(model_path);

    LSTM lstm2 = LSTM::load(model_path);

    std::tuple output = test(lstm, x_train, y_train_chars, idx_to_char);
    std::tuple output2 = test(lstm2, x_train, y_train_chars, idx_to_char);

    // std::string output_str = std::get<0>(output);
    float accuracy = std::get<1>(output);
    float accuracy2 = std::get<1>(output2);

    std::cout << "Test accuracy: " << accuracy << "%" << std::endl;
    std::cout << "Test accuracy 2: " << accuracy2 << "%" << std::endl;
    // std::cout << "Output: " << output_str << std::endl;

    return 0;
}