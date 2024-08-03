#include <fstream>
#include <iostream>
#include <string>

#include "matrix.hpp"
#include <set>
#include <unordered_map>

std::optional<std::string>
read_to_string(const std::string &path) {
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

std::set<char> create_vocab(const std::string &data) {
    return std::set<char>(data.begin(), data.end());
}

int main() {
    std::string dataset_path = "../../datasets/shakespeare/tiny_shakespeare_small.txt";
    std::optional<std::string> maybe_dataset = read_to_string(dataset_path);

    if (!maybe_dataset.has_value()) {
        std::cerr << "Failed to read file: " << dataset_path << std::endl;
        return 1;
    }

    std::string dataset = maybe_dataset.value();
    std::size_t dataset_size = dataset.size();
    std::set<char> vocab = create_vocab(dataset);
    std::size_t vocab_size = vocab.size();

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

    std::string x_train = std::string(dataset.begin(), dataset.end() - 1);
    std::string y_train = std::string(dataset.begin() + 1, dataset.end());

    return 0;
}