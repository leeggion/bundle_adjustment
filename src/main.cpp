/**
 * @file main.cpp
 * @brief Main file for the program
 * @description Сейчас чистый прототип, что примерно будет
 */

#include "utilities/DatasetLoader.hpp"
#include "utilities/Structs.hpp"
#include "utilities/math.hpp"

int main(int argc, char* argv[]) {
    if (argc == 1) {
        std::cout << "No arguments provided\n";
        return 1;
    }
    std::string dataset_path = argv[1];
    DatasetLoader loader(dataset_path);
    if (loader.load())
        std::cout << "Dataset loaded without problems\n";
    else {
        std::cout << "Dataset was not loaded\n";
        return 0;
    }
    return 0;
}
