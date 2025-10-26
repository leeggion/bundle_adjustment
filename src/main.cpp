/**
 * @file main.cpp
 * @brief Main file for the program
 * @description Сейчас чистый прототип, что примерно будет
 */
#include "ba/dataset_loader.h"
#include "ba/gauss_newton.h"
#include "ba/levenberg_marquardt.h"

int main() {
    DatasetLoader loader("data/");
    auto [cameras, points, observations] = loader.load();

    GaussNewtonOptimizer gn;
    gn.optimize(cameras, points, observations);

    LevenbergMarquardtOptimizer lm;
    lm.optimize(cameras, points, observations);

    return 0;
}
