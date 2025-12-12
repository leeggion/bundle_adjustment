#ifndef _t33nsy_BUNDLEADJUSTER
#define _t33nsy_BUNDLEADJUSTER
#define LM_TRIES 100

#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Structs.hpp"

class BundleAdjuster {
   public:
    BundleAdjuster(
        const std::vector<Observation>& obs,
        const std::vector<std::unique_ptr<ceres::CostFunction>>& cost_functions,
        std::vector<std::array<double, 9>>& camera_params,
        std::vector<std::array<double, 3>>& point_params)
        : obs(obs),
          cost_functions(cost_functions),
          camera_params(camera_params),
          point_params(point_params) {}

    /**
     * @brief Вычисляет общую функцию стоимости как половину
     * суммы квадратов ошибок для всех наблюдений.
     *
     * @note Вызов `cost_functions[i]->Evaluate` используется только для
     * получения остатков; производные (якобианы) не вычисляются (передается
     * `nullptr`).
     *
     * @return double Общее значение функции стоимости (0.5 * Total Squared
     * Error).
     */
    auto calculate_total_error() -> double;

    /**
     * @brief Запускает итерационный процесс оптимизации
     * Левенберга-Марквардта (Levenberg-Marquardt) с использованием
     * разреженного дополнения Шура.
     *
     * @param max_iterations Максимальное количество итераций оптимизации.
     * @param initial_lambda Начальное значение параметра демпфирования
     * Лямбда.
     *
     * @return double Финальное значение ошибки после оптимизации.
     */
    auto SolveLM(int max_iterations = 50, double initial_lambda = 1e-4)
        -> double;

    /**
     * @brief Запускает итерационный процесс оптимизации
     * Гаусса-Ньютона (Gauss-Newthon) с использованием
     * разреженного дополнения Шура.
     *
     * @param max_iterations Максимальное количество итераций оптимизации.
     * Лямбда.
     *
     * @return double Финальное значение ошибки после оптимизации.
     */
    auto SolveGN(int max_iterations = 50) -> double;

    /**
     * @brief Экспортирует оптимизированные параметры камер и точек в файлы.
     *
     * @param out_cams Имя файла для экспорта параметров камер.
     * @param out_points Имя файла для экспорта параметров точек.
     *
     * @note Файлы экспортируются в формате TXT, где каждая строка содержит
     * координаты точек (x, y, z в файле с точками) и параметры камер (wx, wy,
     * wz, tx, ty, tz, f, k1, k2 в файле с камерами)
     */
    void Export(std::string out_cams = "output_cameras.txt",
                std::string out_points = "output_points.txt");

    // Вспомогательные API
    void FreezeIntrinsics();    // фиксируем f,k1,k2
    void UnfreezeIntrinsics();  // размораживаем
    void FreezeEverythingExceptExtrinsics();

    // Решение полной системы напрямую (без редукции Шура)
    void SolveFullSystemDirect(
        double lambda, const std::vector<Eigen::Matrix<double, 9, 9>>& U_blocks,
        const std::vector<Eigen::Matrix<double, 3, 3>>& V_blocks,
        const std::vector<Eigen::Vector<double, 9>>& g_c_blocks,
        const std::vector<Eigen::Vector<double, 3>>& g_p_blocks,
        const std::vector<Eigen::Matrix<double, 9, 3>>& W_blocks,
        Eigen::VectorXd& delta_c, std::vector<Eigen::Vector3d>& delta_p_blocks,
        double& delta_c_norm, double& delta_p_norm);

    auto SolveDirect(int max_iterations = 50, double initial_lambda = 1e-3)
        -> double;

   private:
    const std::vector<Observation>& obs;
    const std::vector<std::unique_ptr<ceres::CostFunction>>& cost_functions;
    std::vector<std::array<double, 9>>& camera_params;
    std::vector<std::array<double, 3>>& point_params;

    // Вспомогательные данные, которые инициализируются один раз:
    int num_cameras = 0;
    int num_points = 0;
    int num_observations = 0;

    // Какие параметры камеры оптимизируются.
    // 0 — параметр фиксирован, 1 — обновляется.
    std::array<bool, 9> camera_param_mask = {true, true, true, true, true,
                                             true, true, true, true};

    // Структуры для индексации наблюдений (для построения системы Шура)
    std::vector<std::vector<size_t>> obs_by_point;
    std::vector<std::vector<size_t>> obs_by_cam;  // опционально

    /**
     * @brief Вычисляет функцию стоимости для заданных пробных параметров.
     *
     * @return double Общее значение функции стоимости (0.5 * SSE).
     */
    auto calculate_total_error_for_params(
        const std::vector<std::array<double, 9>>& temp_camera_params,
        const std::vector<std::array<double, 3>>& temp_point_params) -> double;

    /**
     * @brief Инициализирует внутренние переменные и структуры индексации.
     */
    void SetupProblem();

    /**
     * @brief Вычисляет блоки Гессиана (U, V, W) и градиенты (g_c, g_p)
     * для текущих параметров.
     */
    void BuildHessianAndGradient(
        std::vector<Eigen::Matrix<double, 9, 9>>& U_blocks,
        std::vector<Eigen::Matrix<double, 3, 3>>& V_blocks,
        std::vector<Eigen::Vector<double, 9>>& g_c_blocks,
        std::vector<Eigen::Vector<double, 3>>& g_p_blocks,
        std::vector<Eigen::Matrix<double, 9, 3>>& W_blocks);

    /**
     * @brief Решает систему Шура (S * delta_c = b) для приращений камеры
     * (delta_c), а затем использует обратную подстановку для приращений точек
     * (delta_p).
     */
    void SolveReducedSystem(
        double lambda, const std::vector<Eigen::Matrix<double, 9, 9>>& U_blocks,
        const std::vector<Eigen::Matrix<double, 3, 3>>& V_blocks,
        const std::vector<Eigen::Vector<double, 9>>& g_c_blocks,
        const std::vector<Eigen::Vector<double, 3>>& g_p_blocks,
        const std::vector<Eigen::Matrix<double, 9, 3>>& W_blocks,
        Eigen::VectorXd& delta_c, std::vector<Eigen::Vector3d>& delta_p_blocks,
        double& delta_c_norm, double& delta_p_norm);

    /**
     * @brief Решает систему для приращений камеры разложением Холецкого
     * (delta_c), а затем использует обратную подстановку для приращений точек
     * (delta_p).
     */
    void SolveFullSystem(
        const std::vector<Eigen::Matrix<double, 9, 9>>& U_blocks,
        const std::vector<Eigen::Matrix<double, 3, 3>>& V_blocks,
        const std::vector<Eigen::Vector<double, 9>>& g_c_blocks,
        const std::vector<Eigen::Vector<double, 3>>& g_p_blocks,
        const std::vector<Eigen::Matrix<double, 9, 3>>& W_blocks,
        Eigen::VectorXd& delta_c, std::vector<Eigen::Vector3d>& delta_p_blocks,
        double& delta_c_norm, double& delta_p_norm);
};

#endif /* _t33nsy_BUNDLEADJUSTER */