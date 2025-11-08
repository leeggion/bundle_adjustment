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
    Camera cam1 = loader.get_cams()[0];
    Point3D point1 = loader.get_points()[0];
    Observation obs1 = loader.get_observations()[0];
    auto proj1 = projection(cam1, point1);
    std::cout << "Test projection function on first:\n";
    std::cout << proj1[0] << " " << proj1[1] << "\n";
    std::cout << obs1.cam_id << " " << obs1.point_id << " " << obs1.u << " "
              << obs1.v << "\n";
    return 0;
}
