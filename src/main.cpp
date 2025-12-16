#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <vector>

#include "utilities/BundleAdjuster.hpp"
#include "utilities/DatasetLoader.hpp"
#include "utilities/Structs.hpp"
#include "utilities/Visualize.hpp"
#include "utilities/math.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <path_to_BAL_dataset> <LM/GN> <[optional]max_iter_size>\n";
        return 1;
    }
    DatasetLoader loader(argv[1]);
    if (!loader.load()) {
        std::cerr << "Failed to load dataset." << std::endl;
        return 1;
    }
    auto cams_struct = loader.get_cams();
    auto points_struct = loader.get_points();
    auto obs = loader.get_observations();
    std::cout << "Dataset loaded: " << cams_struct.size() << " cameras, "
              << points_struct.size() << " points, " << obs.size()
              << " observations." << std::endl;

    // --- Шаг 1: Подготовка данных ---
    std::vector<std::array<double, 3>> point_params(points_struct.size());
    for (size_t i = 0; i < points_struct.size(); ++i) {
        point_params[i] = {points_struct[i].pos[0], points_struct[i].pos[1],
                           points_struct[i].pos[2]};
    }

    std::vector<std::array<double, 9>> camera_params(cams_struct.size());
    for (size_t i = 0; i < cams_struct.size(); ++i) {
        camera_params[i] = {cams_struct[i].rotation[0],
                            cams_struct[i].rotation[1],
                            cams_struct[i].rotation[2],
                            cams_struct[i].translation[0],
                            cams_struct[i].translation[1],
                            cams_struct[i].translation[2],
                            cams_struct[i].f,
                            cams_struct[i].k1,
                            cams_struct[i].k2};
    }

    // --- Шаг 2: Создание функторов ---
    std::vector<std::unique_ptr<ceres::CostFunction>> cost_functions;
    cost_functions.reserve(obs.size());
    for (const auto& o : obs) {
        cost_functions.push_back(std::unique_ptr<ceres::CostFunction>(
            BALReprojectionError::Create(o.u, o.v)));
    }

    int max_iter_size = 50;
    if (argc > 3) {
        max_iter_size = std::stoi(argv[3]);
    }

    BundleAdjuster adj(obs, cost_functions, camera_params, point_params);

    // --- Этап 1: Полная оптимизация (extrinsics + intrinsics + points)
    std::cout << "\n====================\n";
    std::cout << "STAGE 1: Full BA (optimize all camera params)\n";
    adj.UnfreezeIntrinsics();
    double final_error = INFINITY;
    if (std::string(argv[2]) == "LM") {
        final_error = adj.SolveLM(max_iter_size / 2);
    } else if (std::string(argv[2]) == "GN") {
        final_error = adj.SolveGN(max_iter_size / 2);
    } else {
        final_error = adj.SolveDirect(max_iter_size / 2);
    }

    // --- Этап 2: Фиксация intrinsics
    std::cout << "\n====================\n";
    std::cout << "STAGE 2: Refinement BA (freeze intrinsics)\n";
    adj.FreezeIntrinsics();

    if (std::string(argv[2]) == "LM") {
        final_error = adj.SolveLM(max_iter_size / 2);
    } else if (std::string(argv[2]) == "GN") {
        final_error = adj.SolveGN(max_iter_size / 2);
    } else {
        final_error = adj.SolveDirect(max_iter_size / 2);
    }

    std::cout << "Optimization finished! Final Error: " << final_error
              << std::endl;
    std::cout << "Showing reprojection visualization..." << std::endl;
    int cams_to_see = std::min((size_t)10, cams_struct.size());
    for (size_t i = 0; i < cams_to_see; ++i)
        visualize_reprojection(cams_struct, points_struct, obs, i);
    return 0;
}