/**
 * @file main.cpp
 * @brief Main file for the program
 * @description Сейчас чистый прототип, что примерно будет
 */

#include "utilities/DatasetLoader.hpp"
#include "utilities/Structs.hpp"
#include "utilities/math.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " path_to_BAL_dataset"
                  << std::endl;
        return 1;
    }

    DatasetLoader loader(argv[1]);
    if (!loader.load()) {
        std::cerr << "Failed to load dataset." << std::endl;
        return 1;
    }

    auto cams = loader.get_cams();
    auto points = loader.get_points();
    auto obs = loader.get_observations();

    ceres::Problem problem;

    // Добавляем все наблюдения
    for (const auto& o : obs) {
        ceres::CostFunction* cost_function =
            BALReprojectionError::Create(o.u, o.v);

        problem.AddResidualBlock(
            cost_function,
            nullptr,  // без robust loss
            cams[o.cam_id]
                .rotation.data(),  // angle-axis + translation + f, k1, k2
            points[o.point_id].pos.data());  // XYZ
    }

    // Настройка оптимизатора
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

    // Сохраняем результаты
    std::ofstream fout("optimized_points.txt");
    for (const auto& p : points) fout << p.pos.transpose() << "\n";

    std::cout << "Optimization finished!" << std::endl;
    return 0;
}

// int main(int argc, char* argv[]) {
//     if (argc == 1) {
//         std::cout << "No arguments provided\n";
//         return 1;
//     }
//     std::string dataset_path = argv[1];
//     DatasetLoader loader(dataset_path);
//     if (loader.load())
//         std::cout << "Dataset loaded without problems\n";
//     else {
//         std::cout << "Dataset was not loaded\n";
//         return 0;
//     }
//     Camera cam1 = loader.get_cams()[0];
//     Point3D point1 = loader.get_points()[0];
//     Observation obs1 = loader.get_observations()[0];
//     auto proj1 = projection(cam1, point1);
//     std::cout << "Test projection function on first:\n";
//     std::cout << proj1[0] << " " << proj1[1] << "\n";
//     std::cout << obs1.cam_id << " " << obs1.point_id << " " << obs1.u << " "
//               << obs1.v << "\n";
//     return 0;
// }
