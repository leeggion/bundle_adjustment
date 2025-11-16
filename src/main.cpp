/**
 * @file main.cpp
 * @brief Main file for the program
 * @description Сейчас чистый прототип, что примерно будет
 */

#include "utilities/DatasetLoader.hpp"
#include "utilities/Structs.hpp"
#include "utilities/Visualize.hpp"
#include "utilities/math.hpp"

#include <Eigen/Dense>
#include <array>
#include <ceres/ceres.h> 
#include <memory>
#include <vector>

double calculate_total_error(
    const std::vector<Observation>& obs,
    const std::vector<std::unique_ptr<ceres::CostFunction>>& cost_functions,
    const std::vector<std::array<double, 9>>& camera_params,
    const std::vector<std::array<double, 3>>& point_params) 
{
    double total_squared_error = 0;
    std::vector<double> residual(2);

    for (size_t i = 0; i < obs.size(); ++i) {
        const auto& o = obs[i];
        double* parameters[] = {
            const_cast<double*>(camera_params[o.cam_id].data()),
            const_cast<double*>(point_params[o.point_id].data())
        };
        
        // Вызываем Evaluate только для получения остатков (residuals)
        cost_functions[i]->Evaluate(parameters, residual.data(), nullptr);
        total_squared_error += residual[0] * residual[0] + residual[1] * residual[1];
    }
    return 0.5 * total_squared_error;
}

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

    auto cams_struct = loader.get_cams();
    auto points_struct = loader.get_points();
    auto obs = loader.get_observations();

    std::cout << "Dataset loaded: " << cams_struct.size() << " cameras, "
              << points_struct.size() << " points, " << obs.size() << " observations."
              << std::endl;

    // --- Шаг 1: Подготовка данных ---
    std::vector<std::array<double, 3>> point_params(points_struct.size());
    for (size_t i = 0; i < points_struct.size(); ++i) {
        point_params[i] = {points_struct[i].pos[0], points_struct[i].pos[1], points_struct[i].pos[2]};
    }

    std::vector<std::array<double, 9>> camera_params(cams_struct.size());
    for (size_t i = 0; i < cams_struct.size(); ++i) {
        camera_params[i] = {
            cams_struct[i].rotation[0], cams_struct[i].rotation[1], cams_struct[i].rotation[2],
            cams_struct[i].translation[0], cams_struct[i].translation[1], cams_struct[i].translation[2],
            cams_struct[i].f, cams_struct[i].k1, cams_struct[i].k2
        };
    }

    
    const int num_observations = obs.size();
    const int num_cameras = camera_params.size();
    const int num_points = point_params.size();

    std::vector<std::vector<size_t>> obs_by_point(num_points);
    // (опционально, но полезно)
    std::vector<std::vector<size_t>> obs_by_cam(num_cameras);

    for (size_t i = 0; i < obs.size(); ++i) {
        obs_by_point[obs[i].point_id].push_back(i);
        obs_by_cam[obs[i].cam_id].push_back(i);
    }

    // --- Шаг 2: Создание функторов ---
    std::vector<std::unique_ptr<ceres::CostFunction>> cost_functions;
    cost_functions.reserve(obs.size());
    for (const auto& o : obs) {
        cost_functions.push_back(
            std::unique_ptr<ceres::CostFunction>(
                BALReprojectionError::Create(o.u, o.v)
            )
        );
    }

   // --- Шаг 3: Цикл оптимизации Levenberg-Marquardt (с Дополнением Шура) ---
    
    // (Константы, которые у вас уже были)
    const int num_cam_params = 9 * num_cameras;
    const int num_point_params = 3 * num_points;

    if (num_cam_params > 15000) {
        std::cerr << "--- WARNING ---" << std::endl;
        std::cerr << "Camera parameters (" << num_cam_params << ") too large for DENSE Schur." << std::endl;
        std::cerr << "This might be slow or crash. Consider using a sparse solver for S." << std::endl;
    }

    double lambda = 1e-4;
    double v = 2.0;
    int max_iterations = 200;

    std::cout << "Starting manual LM Optimization (SPARSE_SCHUR)..." << std::endl;
    std::cout << "Camera params: " << num_cam_params << std::endl;
    std::cout << "Point params:  " << num_point_params << std::endl;

    double current_error = calculate_total_error(obs, cost_functions, camera_params, point_params);
    std::cout << "Initial Error: " << current_error << std::endl;

    // ----- Переменные, которые будут обновляться в цикле -----
    // (Блоки Гессиана и градиенты)
    std::vector<Eigen::Matrix<double, 9, 9>> U_blocks(num_cameras);
    std::vector<Eigen::Matrix<double, 3, 3>> V_blocks(num_points);
    std::vector<Eigen::Vector<double, 9>> g_c_blocks(num_cameras);
    std::vector<Eigen::Vector<double, 3>> g_p_blocks(num_points);
    // (Блок W, по одному на каждое наблюдение)
    std::vector<Eigen::Matrix<double, 9, 3>> W_blocks(num_observations);
    // --------------------------------------------------------

    for (int iter = 0; iter < max_iterations; ++iter) {
        
        // 1. Построение H и g (поблочно)
        // Обнуляем аккумуляторы
        for (auto& m : U_blocks) m.setZero();
        for (auto& m : V_blocks) m.setZero();
        for (auto& v : g_c_blocks) v.setZero();
        for (auto& v : g_p_blocks) v.setZero();
        // W_blocks будет просто перезаписан

        for (size_t i = 0; i < obs.size(); ++i) {
            const auto& o = obs[i];
            
            double* cam_ptr = camera_params[o.cam_id].data();
            double* pt_ptr = point_params[o.point_id].data();
            double* parameters[] = {cam_ptr, pt_ptr};

            double residual[2];
            double jacobian_cam_data[2 * 9];
            double jacobian_pt_data[2 * 3];
            double* jacobians[] = {jacobian_cam_data, jacobian_pt_data};

            cost_functions[i]->Evaluate(parameters, residual, jacobians);

            Eigen::Map<Eigen::Vector2d> r_i(residual);
            Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J_c(jacobian_cam_data);
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_p(jacobian_pt_data);

            // J = [A | B] -> J_c = A, J_p = B
            // H = J^T * J = [ A^T*A | A^T*B ] = [ U | W ]
            //               [ B^T*A | B^T*B ]   [ W^T | V ]
            // g = J^T * r = [ A^T*r ] = [ g_c ]
            //               [ B^T*r ]   [ g_p ]

            U_blocks[o.cam_id] += J_c.transpose() * J_c;
            V_blocks[o.point_id] += J_p.transpose() * J_p;
            g_c_blocks[o.cam_id] += J_c.transpose() * r_i;
            g_p_blocks[o.point_id] += J_p.transpose() * r_i;
            W_blocks[i] = J_c.transpose() * J_p; // W для этого *конкретного* наблюдения
        }
        
        bool step_accepted = false;
        for (int lm_trial = 0; lm_trial < 10; ++lm_trial) {
            
            // 2. Построение УМЕНЬШЕННОЙ системы (S * delta_c = b)
            
            // S = (U + lambda*I) - W * (V + lambda*I)^-1 * W^T
            // b = -g_c + W * (V + lambda*I)^-1 * g_p
            
            Eigen::MatrixXd S(num_cam_params, num_cam_params);
            Eigen::VectorXd b(num_cam_params);
            S.setZero();
            
            // 2a. Обработка V и V_inv * g_p
            // (V - блочно-диагональная, инвертируем поблочно)
            std::vector<Eigen::Matrix<double, 3, 3>> V_inv_blocks(num_points);
            std::vector<Eigen::Vector3d> V_inv_g_p_blocks(num_points);

            for (size_t p = 0; p < num_points; ++p) {
                Eigen::Matrix<double, 3, 3> V_damped = V_blocks[p];
                V_damped.diagonal().array() += lambda;
                
                V_inv_blocks[p] = V_damped.inverse(); // 3x3 инверсия - это быстро
                V_inv_g_p_blocks[p] = V_inv_blocks[p] * g_p_blocks[p];
            }

            // 2b. Обработка U и g_c
            for (size_t c = 0; c < num_cameras; ++c) {
                Eigen::Matrix<double, 9, 9> U_damped = U_blocks[c];
                U_damped.diagonal().array() += lambda;
                
                int c_start = c * 9;
                S.block<9, 9>(c_start, c_start) = U_damped; // Диагональные блоки S
                b.segment<9>(c_start) = -g_c_blocks[c];     // Инициализация b
            }

            // 2c. Построение S и b (самая сложная часть)
            // Итерируем по ТОЧКАМ (т.к. каждая точка связывает камеры)
            for (size_t p = 0; p < num_points; ++p) {
                const auto& obs_for_this_point = obs_by_point[p];
                if (obs_for_this_point.empty()) continue;

                const auto& V_inv_p = V_inv_blocks[p];
                const auto& V_inv_g_p = V_inv_g_p_blocks[p];

                // Каждая точка `p` создает "клику" (clique) в S
                // между всеми камерами, которые ее видят.
                for (size_t i_idx = 0; i_idx < obs_for_this_point.size(); ++i_idx) {
                    size_t obs_i_idx = obs_for_this_point[i_idx];
                    int c_i = obs[obs_i_idx].cam_id;
                    int c_i_start = c_i * 9;
                    const auto& W_i = W_blocks[obs_i_idx]; // W_i = J_ci^T * J_p

                    // b = -g_c + W * (V_inv * g_p)
                    b.segment<9>(c_i_start) += W_i * V_inv_g_p;

                    // S = U - W * V_inv * W^T
                    Eigen::Matrix<double, 9, 3> W_V_inv = W_i * V_inv_p;

                    for (size_t j_idx = i_idx; j_idx < obs_for_this_point.size(); ++j_idx) {
                        size_t obs_j_idx = obs_for_this_point[j_idx];
                        int c_j = obs[obs_j_idx].cam_id;
                        int c_j_start = c_j * 9;
                        const auto& W_j = W_blocks[obs_j_idx]; // W_j = J_cj^T * J_p

                        // S_ij = - W_i * V_inv * W_j^T
                        Eigen::Matrix<double, 9, 9> S_ij = W_V_inv * W_j.transpose();

                        if (c_i == c_j) {
                            // Диагональный блок (i == j)
                            S.block<9, 9>(c_i_start, c_i_start) -= S_ij;
                        } else {
                            // Внедиагональные блоки
                            S.block<9, 9>(c_i_start, c_j_start) -= S_ij;
                            S.block<9, 9>(c_j_start, c_i_start) -= S_ij.transpose();
                        }
                    }
                }
            }

            // 3. Решение УМЕНЬШЕННОЙ системы (для камер)
            // S * delta_c = b
            Eigen::VectorXd delta_c = S.ldlt().solve(b);

            if (delta_c.norm() < 1e-8) {
                std::cout << "Convergence reached (delta_c too small)." << std::endl;
                iter = max_iterations;
                break;
            }
            
            // 4. Обратная подстановка (Back-substitution) для точек
            // delta_p = (V + lambda*I)^-1 * (-g_p - W^T * delta_c)
            std::vector<Eigen::Vector3d> delta_p_blocks(num_points);
            
            // Сначала delta_p = -g_p
            for(size_t p=0; p < num_points; ++p) {
                delta_p_blocks[p] = -g_p_blocks[p];
            }

            // Затем delta_p -= W^T * delta_c (по всем наблюдениям)
            for (size_t i = 0; i < obs.size(); ++i) {
                const auto& o = obs[i];
                const auto& W_i = W_blocks[i]; // W_i = J_c^T * J_p
                // W_i^T = J_p^T * J_c
                Eigen::Vector<double, 9> dc = delta_c.segment<9>(o.cam_id * 9);
                delta_p_blocks[o.point_id] -= W_i.transpose() * dc;
            }

            // Финально delta_p = V_inv * delta_p
            for (size_t p = 0; p < num_points; ++p) {
                delta_p_blocks[p] = V_inv_blocks[p] * delta_p_blocks[p];
            }
            
            // 5. Пробный шаг: создаем новые параметры
            std::vector<std::array<double, 9>> new_camera_params = camera_params;
            std::vector<std::array<double, 3>> new_point_params = point_params;

            for (size_t c = 0; c < num_cameras; ++c) {
                for (int j = 0; j < 9; ++j) new_camera_params[c][j] += delta_c[c * 9 + j];
            }
            for (size_t p = 0; p < num_points; ++p) {
                for (int j = 0; j < 3; ++j) new_point_params[p][j] += delta_p_blocks[p][j];
            }

            // 6. Оцениваем ошибку с новыми параметрами
            double new_error = calculate_total_error(obs, cost_functions, new_camera_params, new_point_params);

            // 7. Логика LM
            if (new_error < current_error) {
                std::cout << "  Iter " << iter << " (trial " << lm_trial << "): SUCCESS, Error: " 
                          << new_error << ", Lambda: " << lambda << std::endl;
                
                lambda = std::max(lambda / 3.0, 1e-7);
                v = 2.0;
                
                camera_params = std::move(new_camera_params);
                point_params = std::move(new_point_params);
                current_error = new_error;
                
                step_accepted = true;
                break; 
            } else {
                std::cout << "  Iter " << iter << " (trial " << lm_trial << "): FAILED,  Error: " 
                          << new_error << ", Lambda: " << lambda << std::endl;

                lambda = lambda * v;
                v = v * 2.0;
                step_accepted = false;
            }
        } // конец цикла попыток LM

        if (!step_accepted) {
            std::cout << "LM failed to find a better step." << std::endl;
            break;
        }
    } // конец главного цикла итераций

    std::cout << "Optimization finished!" << std::endl;

    // --- Шаг 4: Сохранение и визуализация ---

    std::cout << "Saving optimized points..." << std::endl;
    std::ofstream fout("optimized_points.txt");
    for (const auto& p : point_params) {
        fout << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
    fout.close();

    std::cout << "Saving optimized cameras..." << std::endl;
    std::ofstream fcams("optimized_cameras.txt");
    for (const auto& c : camera_params) {
        fcams << c[0] << " " << c[1] << " " << c[2] << " "
              << c[3] << " " << c[4] << " " << c[5] << " "
              << c[6] << " " << c[7] << " " << c[8] << "\n";
    }
    fcams.close();

    // Копируем данные обратно в исходные структуры для визуализации
    for (size_t i = 0; i < cams_struct.size(); ++i) {
        cams_struct[i].rotation    = Eigen::Vector3d(camera_params[i].data());
        cams_struct[i].translation = Eigen::Vector3d(camera_params[i].data() + 3);
        cams_struct[i].f           = camera_params[i][6];
        cams_struct[i].k1          = camera_params[i][7];
        cams_struct[i].k2          = camera_params[i][8];
    }
    for (size_t i = 0; i < points_struct.size(); ++i) {
        points_struct[i].pos = Eigen::Vector3d(point_params[i].data());
    }

    std::cout << "Showing reprojection visualization..." << std::endl;
    int cams_to_see = std::min((size_t)10, cams_struct.size());
    for (size_t i = 0; i < cams_to_see; ++i)
        visualize_reprojection(cams_struct, points_struct, obs, i);

    return 0;
}