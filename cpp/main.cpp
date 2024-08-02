#include <iostream>

#include "matrix.hpp"

int main() {
    Matrix mat = Matrix::uniform(3, 3, 0.0f, 1.0f);
    std::cout << mat << std::endl;
    return 0;
}