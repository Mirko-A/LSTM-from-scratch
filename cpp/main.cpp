#include <fstream>
#include <iostream>
#include <string>

#include "matrix.hpp"
#include <set>

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
        if (file.peek() != EOF) {
            content.push_back('\n');
        }
    }

    return std::make_optional(content);
}

std::set<char> create_vocab(const std::string &data) {
    return std::set<char>(data.begin(), data.end());
}

int main() {
    std::string dataset_path = "../../datasets/shakespeare/tiny_shakespeare_small.txt";
    std::optional<std::string> maybe_content = read_to_string(dataset_path);

    if (!maybe_content.has_value()) {
        std::cerr << "Failed to read file: " << dataset_path << std::endl;
        return 1;
    }

    std::string content = maybe_content.value();
    std::set<char> vocab = create_vocab(content);

    for (char c : vocab) {
        std::cout << c << " ";
    }

    std::cout << std::endl;

    return 0;
}