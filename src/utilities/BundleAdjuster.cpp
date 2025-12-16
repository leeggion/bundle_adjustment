#include "BundleAdjuster.hpp"

auto BundleAdjuster::calculate_total_error() -> double {
    double total_squared_error = 0;
    std::vector<double> residual(2);
    for (size_t i = 0; i < obs.size(); ++i) {
        const auto& o = obs[i];
        double* parameters[] = {
            const_cast<double*>(camera_params[o.cam_id].data()),
            const_cast<double*>(point_params[o.point_id].data())};
        cost_functions[i]->Evaluate(parameters, residual.data(), nullptr);
        total_squared_error +=
            residual[0] * residual[0] + residual[1] * residual[1];
    }
    return 0.5 * total_squared_error;
}

auto BundleAdjuster::calculate_total_error_for_params(
    const std::vector<std::array<double, 9>>& temp_camera_params,
    const std::vector<std::array<double, 3>>& temp_point_params) -> double {
    double total_squared_error = 0;
    std::vector<double> residual(2);
    for (size_t i = 0; i < obs.size(); ++i) {
        const auto& o = obs[i];
        double* parameters[] = {
            const_cast<double*>(temp_camera_params[o.cam_id].data()),
            const_cast<double*>(temp_point_params[o.point_id].data())};
        // Используем cost_functions члена класса
        cost_functions[i]->Evaluate(parameters, residual.data(), nullptr);
        total_squared_error +=
            residual[0] * residual[0] + residual[1] * residual[1];
    }
    return 0.5 * total_squared_error;
}

auto BundleAdjuster::SolveLM(int max_iterations, double initial_lambda)
    -> double {
    SetupProblem();
    double lambda = initial_lambda;
    double v = 2.0;
    double current_error = calculate_total_error();

    std::cout << "=================================\n"
              << "LM optimizer started\n"
              << "\tCameras: " << this->num_cameras
              << "\n\tPoints: " << this->num_points << "\n"
              << "Initial Error: " << current_error << std::endl;

    // Поблочные матрицы
    std::vector<Eigen::Matrix<double, 9, 9>> U_blocks(num_cameras);
    std::vector<Eigen::Matrix<double, 3, 3>> V_blocks(num_points);
    std::vector<Eigen::Vector<double, 9>> g_c_blocks(num_cameras);
    std::vector<Eigen::Vector<double, 3>> g_p_blocks(num_points);
    std::vector<Eigen::Matrix<double, 9, 3>> W_blocks(num_observations);

    auto start = std::chrono::system_clock::now();

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::cout << "\n[ITERATION " << iter << "]" << std::endl;
        std::cout << "Current Error: " << current_error << std::endl;

        // 1) Построение гессиана и градиента
        BuildHessianAndGradient(U_blocks, V_blocks, g_c_blocks, g_p_blocks,
                                W_blocks);

        bool step_accepted = false;

        for (int lm_try = 0; lm_try < LM_TRIES; ++lm_try) {
            std::cout << "  [LM Trial " << lm_try << "] Lambda: " << lambda
                      << ", v: " << v << std::endl;

            Eigen::VectorXd delta_c;
            std::vector<Eigen::Vector3d> delta_p_blocks;
            double delta_c_norm = 0.0;
            double delta_p_norm = 0.0;

            // 2) Решение редуцированной системы
            SolveReducedSystem(lambda, U_blocks, V_blocks, g_c_blocks,
                               g_p_blocks, W_blocks, delta_c, delta_p_blocks,
                               delta_c_norm, delta_p_norm);

            // Проверка корректности размеров
            if (delta_c.size() == 0) {
                std::cout
                    << "[ERROR] LM solver returned empty delta_c. Aborting."
                    << std::endl;
                return current_error;
            }
            if ((size_t)delta_c.size() < num_cameras * 9) {
                std::cout << "[WARNING] delta_c too small: " << delta_c.size()
                          << " < expected " << num_cameras * 9 << std::endl;
            }
            if (delta_p_blocks.size() < num_points) {
                std::cout << "[WARNING] delta_p_blocks too small: "
                          << delta_p_blocks.size() << " < expected "
                          << num_points << std::endl;
            }

            // Если нормы не заполнены — считаем
            if (delta_c_norm <= 0.0) delta_c_norm = delta_c.norm();
            if (delta_p_norm <= 0.0 && delta_p_blocks.size() > 0) {
                double acc = 0.0;
                for (size_t i = 0;
                     i < std::min(delta_p_blocks.size(), (size_t)num_points);
                     ++i)
                    acc += delta_p_blocks[i].squaredNorm();
                delta_p_norm = std::sqrt(acc);
            }

            std::cout << "    | Step norms: ||Delta_c|| = " << delta_c_norm
                      << ", ||Delta_p|| = " << delta_p_norm << std::endl;

            // Проверка сходимости
            if (delta_c_norm < 1e-8) {
                std::cout << "[CONVERGENCE] Delta_c norm too small. Finishing."
                          << std::endl;
                return current_error;
            }

            // 3) Пробный шаг
            std::vector<std::array<double, 9>> new_camera_params =
                camera_params;
            std::vector<std::array<double, 3>> new_point_params = point_params;

            // Применяем delta_c
            for (size_t c = 0; c < num_cameras; ++c) {
                for (int j = 0; j < 9; ++j) {
                    size_t idx = c * 9 + j;
                    if (idx < (size_t)delta_c.size())
                        if (camera_param_mask[j] &&
                            idx < (size_t)delta_c.size())
                            new_camera_params[c][j] +=
                                delta_c(static_cast<int>(idx));
                    // new_camera_params[c][j] += delta_c((int)idx);
                }
            }

            // Применяем delta_p
            for (size_t p = 0; p < num_points; ++p) {
                if (p < delta_p_blocks.size()) {
                    for (int j = 0; j < 3; ++j)
                        new_point_params[p][j] += delta_p_blocks[p][j];
                }
            }

            // 4) Считаем ошибку
            double trial_error = calculate_total_error_for_params(
                new_camera_params, new_point_params);
            std::cout << "    | Trial Error: " << trial_error << std::endl;

            // 5) Логика LM
            if (trial_error < current_error) {
                // Step ACCEPTED
                std::cout << "    | SUCCESS" << std::endl;

                camera_params = std::move(new_camera_params);
                point_params = std::move(new_point_params);
                current_error = trial_error;
                step_accepted = true;

                lambda = std::max(lambda / 3.0, 1e-7);
                v = 2.0;
                break;
            } else {
                // Step REJECTED
                std::cout << "    | FAILED" << std::endl;
                lambda *= v;
                v *= 2.0;
                step_accepted = false;
            }
        }

        if (!step_accepted) {
            std::cout << "[HALT] LM failed to find a valid step. Stopping."
                      << std::endl;
            break;
        }
    }

    auto end = std::chrono::system_clock::now();
    std::cout << "LM time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms\n";
    this->Export();
    return current_error;
}

auto BundleAdjuster::SolveGN(int max_iterations) -> double {
    SetupProblem();
    double current_error = calculate_total_error();
    std::cout << "=================================\n"
              << "GN optimizer started\n"
              << "\tCameras: " << this->num_cameras
              << "\n\tPoints: " << this->num_points << "\n"
              << "Initial Error: " << current_error << std::endl;

    // Поблочные матрицы/векторы, которые будут перезаполняться в каждой
    // итерации
    std::vector<Eigen::Matrix<double, 9, 9>> U_blocks(num_cameras);
    std::vector<Eigen::Matrix<double, 3, 3>> V_blocks(num_points);
    std::vector<Eigen::Vector<double, 9>> g_c_blocks(num_cameras);
    std::vector<Eigen::Vector<double, 3>> g_p_blocks(num_points);
    std::vector<Eigen::Matrix<double, 9, 3>> W_blocks(num_observations);

    auto start = std::chrono::system_clock::now();

    bool warn_once =
        false;  // флаг, показывающий, что было одно увеличение ошибки

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::cout << "\n[ITERATION " << iter << "]" << std::endl;
        std::cout << "Current Error: " << current_error << std::endl;

        // 1) Построение H и g (поблочно)
        BuildHessianAndGradient(U_blocks, V_blocks, g_c_blocks, g_p_blocks,
                                W_blocks);

        // 2) Решение уменьшенной системы и обратная подстановка
        Eigen::VectorXd delta_c;                      // размер: num_cameras * 9
        std::vector<Eigen::Vector3d> delta_p_blocks;  // size: num_points

        // Инициализируем нормы на случай, если решатель не заполнит выходные
        double delta_c_norm = 0.0;
        double delta_p_norm = 0.0;

        SolveReducedSystem(0.0, U_blocks, V_blocks, g_c_blocks, g_p_blocks,
                           W_blocks, delta_c, delta_p_blocks, delta_c_norm,
                           delta_p_norm);

        // Проверка размеров и корректности выходных данных
        if (delta_c.size() == 0) {
            std::cout << "[ERROR] Solver returned empty delta_c. Terminating."
                      << std::endl;
            break;
        }
        if (static_cast<size_t>(delta_c.size()) < num_cameras * 9) {
            std::cout << "[WARNING] delta_c has unexpected size: "
                      << delta_c.size() << " (expected at least "
                      << num_cameras * 9 << ")." << std::endl;
            // продолжаем, но будем аккуратно индексировать
        }
        if (delta_p_blocks.size() < num_points) {
            std::cout << "[WARNING] delta_p_blocks has unexpected size: "
                      << delta_p_blocks.size() << " (expected " << num_points
                      << ")." << std::endl;
        }

        // Если решатель не заполнил нормы, посчитаем их явно
        if (delta_c_norm <= 0.0) delta_c_norm = delta_c.norm();
        if (delta_p_blocks.size() > 0 && delta_p_norm <= 0.0) {
            double acc = 0.0;
            for (size_t i = 0;
                 i < std::min(delta_p_blocks.size(), (size_t)num_points); ++i)
                acc += delta_p_blocks[i].squaredNorm();
            delta_p_norm = std::sqrt(acc);
        }

        std::cout << "    | Step norms: ||Delta_c|| = " << delta_c_norm
                  << ", ||Delta_p|| = " << delta_p_norm << std::endl;

        // 3) Проверка сходимости
        const double tol = 1e-8;
        if (delta_c_norm < tol) {
            std::cout << "\n[CONVERGENCE] Delta_c norm (" << delta_c_norm
                      << ") < " << tol << ". Optimization finished."
                      << std::endl;
            break;
        }

        // 4) Пробный шаг: создаем новые параметры и аккуратно применяем
        // приращения
        std::vector<std::array<double, 9>> new_camera_params = camera_params;
        std::vector<std::array<double, 3>> new_point_params = point_params;

        // Применение приращений к камерам — внимательно проверяем границы
        size_t delta_c_expected = static_cast<size_t>(num_cameras * 9);
        for (size_t c = 0; c < num_cameras; ++c) {
            for (int j = 0; j < 9; ++j) {
                size_t idx = c * 9 + j;
                if (idx < static_cast<size_t>(delta_c.size())) {
                    if (camera_param_mask[j] && idx < (size_t)delta_c.size())
                        new_camera_params[c][j] +=
                            delta_c(static_cast<int>(idx));
                    // new_camera_params[c][j] +=
                    // delta_c(static_cast<int>(idx));
                } else {
                    // если delta_c короче, считаем недостающие приращения
                    // нулями
                }
            }
        }

        // Применение приращений к точкам — также проверяем границы
        for (size_t p = 0; p < num_points; ++p) {
            if (p < delta_p_blocks.size()) {
                for (int j = 0; j < 3; ++j)
                    new_point_params[p][j] += delta_p_blocks[p][j];
            } else {
                // недостающие приращения считаются нулями
            }
        }

        // 5) Оцениваем ошибку с новыми параметрами
        double new_error = calculate_total_error_for_params(new_camera_params,
                                                            new_point_params);
        std::cout << "  | New Error: " << new_error << std::endl;

        // 6) Проверка улучшения (для отладки/защиты стабильности)
        if (new_error > current_error) {
            if (!warn_once) {
                std::cout << "  [WARNING] Error increased. Marking warning and "
                             "not accepting step."
                          << std::endl;
                // не принимаем шаг — оставляем параметры прежними, но разрешаем
                // одной попытке повториться
                warn_once = true;
                continue;  // следующая итерация: пересчитает H и попробует
                           // снова
            } else {
                std::cout << "  [WARNING] Error increased again. Stopping GN."
                          << std::endl;
                break;
            }
        }

        // 7) Принятие шага
        camera_params = std::move(new_camera_params);
        point_params = std::move(new_point_params);
        current_error = new_error;

        // сбрасываем флаг предупреждения, т.к. шаг принят
        warn_once = false;
    }

    auto end = std::chrono::system_clock::now();
    std::cout << "GN time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms\n";
    this->Export();
    return current_error;
}

void BundleAdjuster::Export(std::string out_cams, std::string out_points) {
    std::cout << "Saving optimized points..." << std::endl;
    std::ofstream fout(out_points);
    for (const auto& p : point_params) {
        fout << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
    fout.close();
    std::cout << "Saving optimized cameras..." << std::endl;
    std::ofstream fcams(out_cams);
    for (const auto& c : camera_params) {
        fcams << c[0] << " " << c[1] << " " << c[2] << " " << c[3] << " "
              << c[4] << " " << c[5] << " " << c[6] << " " << c[7] << " "
              << c[8] << "\n";
    }
    fcams.close();
}

void BundleAdjuster::FreezeIntrinsics() {
    camera_param_mask[6] = false;  // f
    camera_param_mask[7] = false;  // k1
    camera_param_mask[8] = false;  // k2
}

void BundleAdjuster::UnfreezeIntrinsics() {
    camera_param_mask[6] = true;
    camera_param_mask[7] = true;
    camera_param_mask[8] = true;
}

void BundleAdjuster::FreezeEverythingExceptExtrinsics() {
    for (int i = 0; i < 9; i++)
        camera_param_mask[i] = (i < 6);  // R,t — разрешены, остальное запретить
}

void BundleAdjuster::SetupProblem() {
    num_cameras = camera_params.size();
    num_points = point_params.size();
    num_observations = obs.size();

    // Инициализация структур для индексации
    obs_by_point.resize(num_points);
    obs_by_cam.resize(num_cameras);

    for (size_t i = 0; i < obs.size(); ++i) {
        obs_by_point[obs[i].point_id].push_back(i);
        obs_by_cam[obs[i].cam_id].push_back(i);
    }
}

void BundleAdjuster::BuildHessianAndGradient(
    std::vector<Eigen::Matrix<double, 9, 9>>& U_blocks,
    std::vector<Eigen::Matrix<double, 3, 3>>& V_blocks,
    std::vector<Eigen::Vector<double, 9>>& g_c_blocks,
    std::vector<Eigen::Vector<double, 3>>& g_p_blocks,
    std::vector<Eigen::Matrix<double, 9, 3>>& W_blocks) {
    // Обнуляем аккумуляторы и убеждаемся, что они правильного размера
    for (auto& m : U_blocks) m.setZero();
    for (auto& m : V_blocks) m.setZero();
    for (auto& v : g_c_blocks) v.setZero();
    for (auto& v : g_p_blocks) v.setZero();
    // W_blocks просто перезаписывается
    for (size_t i = 0; i < obs.size(); ++i) {
        const auto& o = obs[i];
        double* cam_ptr = camera_params[o.cam_id].data();
        double* pt_ptr = point_params[o.point_id].data();
        double* parameters[] = {cam_ptr, pt_ptr};
        double residual[2];
        double jacobian_cam_data[2 * 9];
        double jacobian_pt_data[2 * 3];
        double* jacobians[] = {jacobian_cam_data, jacobian_pt_data};
        // Вызываем Evaluate для получения остатков И якобианов
        cost_functions[i]->Evaluate(parameters, residual, jacobians);
        // Использование Eigen::Map для удобной работы с данными
        Eigen::Map<Eigen::Vector2d> r_i(residual);
        // RowMajor т.к. Ceres записывает данные построчно
        Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J_c(
            jacobian_cam_data);
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_p(
            jacobian_pt_data);
        // Накопление блоков Гессиана (H = J^T * J)
        // U (диагональный блок камеры): U = A^T * A
        U_blocks[o.cam_id] += J_c.transpose() * J_c;
        // V (диагональный блок точки): V = B^T * B
        V_blocks[o.point_id] += J_p.transpose() * J_p;
        // Внедиагональный блок: W = A^T * B
        W_blocks[i] = J_c.transpose() * J_p;
        // Накопление градиентов (g = J^T * r)
        // g_c (градиент камеры): g_c = A^T * r
        g_c_blocks[o.cam_id] += J_c.transpose() * r_i;
        // g_p (градиент точки): g_p = B^T * r
        g_p_blocks[o.point_id] += J_p.transpose() * r_i;
    }
}

void BundleAdjuster::SolveReducedSystem(
    double lambda, const std::vector<Eigen::Matrix<double, 9, 9>>& U_blocks,
    const std::vector<Eigen::Matrix<double, 3, 3>>& V_blocks,
    const std::vector<Eigen::Vector<double, 9>>& g_c_blocks,
    const std::vector<Eigen::Vector<double, 3>>& g_p_blocks,
    const std::vector<Eigen::Matrix<double, 9, 3>>& W_blocks,
    Eigen::VectorXd& delta_c, std::vector<Eigen::Vector3d>& delta_p_blocks,
    double& delta_c_norm, double& delta_p_norm) {
    const int num_cam_params = 9 * num_cameras;
    // Инициализация матриц и векторов для редуцированной системы
    Eigen::MatrixXd S(num_cam_params, num_cam_params);
    Eigen::VectorXd b(num_cam_params);
    S.setZero();
    b.setZero();  // Сначала установим в 0, а потом добавим -g_c
    // 1. Предварительный расчет V_inv и V_inv * g_p (для каждой точки)
    std::vector<Eigen::Matrix<double, 3, 3>> V_inv_blocks(num_points);
    std::vector<Eigen::Vector3d> V_inv_g_p_blocks(num_points);
    delta_p_blocks.resize(num_points);
    for (size_t p = 0; p < num_points; ++p) {
        Eigen::Matrix<double, 3, 3> V_damped = V_blocks[p];
        // Демпфирование V: V + lambda*I
        V_damped.diagonal().array() += lambda;
        // 3x3 инверсия (быстро)
        V_inv_blocks[p] = V_damped.inverse();
        V_inv_g_p_blocks[p] = V_inv_blocks[p] * g_p_blocks[p];
        // Инициализация delta_p для обратной подстановки: delta_p = -g_p ...
        delta_p_blocks[p] = -g_p_blocks[p];
    }
    // 2. Построение S и b
    for (size_t c = 0; c < num_cameras; ++c) {
        Eigen::Matrix<double, 9, 9> U_damped = U_blocks[c];
        // Демпфирование U: U + lambda*I
        U_damped.diagonal().array() += lambda;

        int c_start = c * 9;
        // Диагональные блоки S: S_cc = U_c + lambda*I
        S.block<9, 9>(c_start, c_start) = U_damped;
        // Инициализация b: b_c = -g_c
        b.segment<9>(c_start) = -g_c_blocks[c];
    }
    // Итерация по точкам для добавления внедиагональных блоков Шура
    // S_ij = S_ij - W_i * (V + lambda*I)^-1 * W_j^T
    // b_i = b_i + W_i * (V + lambda*I)^-1 * g_p
    for (size_t p = 0; p < num_points; ++p) {
        const auto& obs_for_this_point = obs_by_point[p];
        if (obs_for_this_point.empty()) continue;
        const auto& V_inv_p = V_inv_blocks[p];
        const auto& V_inv_g_p = V_inv_g_p_blocks[p];
        // Построение клики для всех камер, видящих точку p
        for (size_t i_idx = 0; i_idx < obs_for_this_point.size(); ++i_idx) {
            size_t obs_i_idx = obs_for_this_point[i_idx];
            int c_i = obs[obs_i_idx].cam_id;
            int c_i_start = c_i * 9;
            const auto& W_i = W_blocks[obs_i_idx];
            // Добавление в вектор b
            b.segment<9>(c_i_start) += W_i * V_inv_g_p;
            // Расчет промежуточного W_i * V_inv
            Eigen::Matrix<double, 9, 3> W_V_inv = W_i * V_inv_p;
            for (size_t j_idx = i_idx; j_idx < obs_for_this_point.size();
                 ++j_idx) {
                size_t obs_j_idx = obs_for_this_point[j_idx];
                int c_j = obs[obs_j_idx].cam_id;
                int c_j_start = c_j * 9;
                const auto& W_j = W_blocks[obs_j_idx];
                // Вычисление S_ij = W_i * V_inv * W_j^T
                Eigen::Matrix<double, 9, 9> S_ij = W_V_inv * W_j.transpose();
                // Обновление матрицы S: S = S - S_ij
                if (c_i == c_j) {
                    // Диагональный блок
                    S.block<9, 9>(c_i_start, c_i_start) -= S_ij;
                } else {
                    // Внедиагональные блоки (используем симметрию)
                    S.block<9, 9>(c_i_start, c_j_start) -= S_ij;
                    S.block<9, 9>(c_j_start, c_i_start) -= S_ij.transpose();
                }
            }
        }
    }
    // 3. Решение редуцированной системы S * delta_c = b
    // Используем LDLT (разложение Холецкого) для симметричной
    // положительно-определенной/полуопределенной матрицы S
    delta_c = S.ldlt().solve(b);
    // 4. Обратная подстановка для delta_p
    // delta_p = (V + lambda*I)^-1 * (-g_p - W^T * delta_c)
    // (-g_p) уже инициализировано в delta_p_blocks
    // Вычисляем W^T * delta_c (по всем наблюдениям)
    for (size_t i = 0; i < obs.size(); ++i) {
        const auto& o = obs[i];
        const auto& W_i = W_blocks[i];  // W_i = J_c^T * J_p

        // Получаем соответствующий приращение камеры
        Eigen::Vector<double, 9> dc = delta_c.segment<9>(o.cam_id * 9);

        // W^T * delta_c = (J_c^T * J_p)^T * delta_c = J_p^T * J_c * delta_c
        // Вычитаем W^T * delta_c
        delta_p_blocks[o.point_id] -= W_i.transpose() * dc;
    }
    // Финально delta_p = V_inv * (полученный вектор)
    for (size_t p = 0; p < num_points; ++p) {
        delta_p_blocks[p] = V_inv_blocks[p] * delta_p_blocks[p];
    }
    // Вычисляем норму вектора приращений камеры
    delta_c_norm = delta_c.norm();
    // Вычисляем норму вектора приращений точек
    double total_delta_p_sq = 0.0;
    for (const auto& dp : delta_p_blocks) {
        total_delta_p_sq += dp.squaredNorm();
    }
    delta_p_norm = std::sqrt(total_delta_p_sq);
}

void BundleAdjuster::SolveFullSystem(
    const std::vector<Eigen::Matrix<double, 9, 9>>& U_blocks,
    const std::vector<Eigen::Matrix<double, 3, 3>>& V_blocks,
    const std::vector<Eigen::Vector<double, 9>>& g_c_blocks,
    const std::vector<Eigen::Vector<double, 3>>& g_p_blocks,
    const std::vector<Eigen::Matrix<double, 9, 3>>& W_blocks,
    Eigen::VectorXd& delta_c, std::vector<Eigen::Vector3d>& delta_p_blocks,
    double& delta_c_norm, double& delta_p_norm) {
    // Общее количество параметров: N_c * 9 + N_p * 3
    const int num_cam_params = 9 * num_cameras;
    const int num_point_params = 3 * num_points;
    const int total_params = num_cam_params + num_point_params;

    // Вектор правой части (градиент)
    Eigen::VectorXd g(total_params);
    // Матрица Гессиана (разреженная)
    // В Bundle Adjustment она симметрична и положительно-полуопределенна
    Eigen::SparseMatrix<double> H(total_params, total_params);
    // Структура для хранения ненулевых элементов
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(num_cam_params * 9 + num_point_params * 3 +
                        num_observations * 2 * 9 * 3);

    // 1. Построение вектора градиента g = [g_c, g_p]^T
    // Часть g_c
    for (size_t c = 0; c < num_cameras; ++c) {
        g.segment<9>(c * 9) = g_c_blocks[c];
    }
    // Часть g_p
    for (size_t p = 0; p < num_points; ++p) {
        g.segment<3>(num_cam_params + p * 3) = g_p_blocks[p];
    }
    // Вектор правой части для нормальных уравнений: -g
    Eigen::VectorXd rhs = -g;

    // 2. Построение разреженной матрицы Гессиана H
    // H = [ U | W ]
    //     [ W^T | V ]

    // A) Диагональный блок U (Камеры)
    for (size_t c = 0; c < num_cameras; ++c) {
        int r_start = c * 9;
        const auto& U = U_blocks[c];
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                tripletList.emplace_back(r_start + i, r_start + j, U(i, j));
            }
        }
    }

    // B) Диагональный блок V (Точки)
    for (size_t p = 0; p < num_points; ++p) {
        int r_start = num_cam_params + p * 3;
        const auto& V = V_blocks[p];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                tripletList.emplace_back(r_start + i, r_start + j, V(i, j));
            }
        }
    }

    // C) Внедиагональные блоки W и W^T (Связи Камера-Точка)
    for (size_t i = 0; i < obs.size(); ++i) {
        const auto& o = obs[i];
        int c_id = o.cam_id;
        int p_id = o.point_id;
        int r_c_start = c_id * 9;
        int c_p_start = num_cam_params + p_id * 3;
        const auto& W = W_blocks[i];  // W: 9x3

        // Блок W (Верхний правый, 9x3)
        // Строки: Камера (c_id), Столбцы: Точка (p_id)
        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 3; ++c) {
                // H(c_i, p_j) = W_ij
                tripletList.emplace_back(r_c_start + r, c_p_start + c, W(r, c));
            }
        }

        // Блок W^T (Нижний левый, 3x9)
        // Строки: Точка (p_id), Столбцы: Камера (c_id)
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 9; ++c) {
                // H(p_j, c_i) = W_ij^T
                tripletList.emplace_back(c_p_start + r, r_c_start + c, W(c, r));
            }
        }
    }

    H.setFromTriplets(tripletList.begin(), tripletList.end());

    // 3. Решение системы H * delta = -g с помощью разреженного Холецкого (LDLT)
    // Разложение Холецкого (SimplicialLDLT) эффективно для разреженных
    // симметричных положительно-определенных/полуопределенных матриц.
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    // Символическое разложение (вычисляется один раз, если структура не
    // меняется) Можно вынести за цикл, если структура H остается неизменной, но
    // в BA структура H меняется из-за различных видимых точек.
    solver.analyzePattern(H);
    // Численное разложение и решение
    solver.factorize(H);

    // Проверка, что разложение прошло успешно
    if (solver.info() != Eigen::Success) {
        std::cerr << "[ERROR] Cholesky factorization failed." << std::endl;
        // Возвращаем нулевое приращение
        delta_c = Eigen::VectorXd::Zero(num_cam_params);
        delta_p_blocks.assign(num_points, Eigen::Vector3d::Zero());
        delta_c_norm = 0.0;
        delta_p_norm = 0.0;
        return;
    }

    Eigen::VectorXd delta = solver.solve(rhs);

    // 4. Разделение решения delta на delta_c и delta_p
    delta_c = delta.head(num_cam_params);
    delta_p_blocks.resize(num_points);
    double total_delta_p_sq = 0.0;

    for (size_t p = 0; p < num_points; ++p) {
        delta_p_blocks[p] = delta.segment<3>(num_cam_params + p * 3);
        total_delta_p_sq += delta_p_blocks[p].squaredNorm();
    }

    // 5. Вычисление норм
    delta_c_norm = delta_c.norm();
    delta_p_norm = std::sqrt(total_delta_p_sq);
}

void BundleAdjuster::SolveFullSystemDirect(
    double lambda, const std::vector<Eigen::Matrix<double, 9, 9>>& U_blocks,
    const std::vector<Eigen::Matrix<double, 3, 3>>& V_blocks,
    const std::vector<Eigen::Vector<double, 9>>& g_c_blocks,
    const std::vector<Eigen::Vector<double, 3>>& g_p_blocks,
    const std::vector<Eigen::Matrix<double, 9, 3>>& W_blocks,
    Eigen::VectorXd& delta_c, std::vector<Eigen::Vector3d>& delta_p_blocks,
    double& delta_c_norm, double& delta_p_norm) {
    const int num_cam_params = 9 * num_cameras;
    const int num_point_params = 3 * num_points;
    const int total_params = num_cam_params + num_point_params;

    // 1. Создаем полную матрицу Гессиана H и вектор градиента g
    Eigen::MatrixXd H(total_params, total_params);
    Eigen::VectorXd g(total_params);
    H.setZero();
    g.setZero();

    // 2. Заполняем диагональные блоки U и g_c
    for (size_t c = 0; c < num_cameras; ++c) {
        int start_idx = static_cast<int>(c) * 9;
        // Добавляем демпфирование Левенберга-Марквардта
        Eigen::Matrix<double, 9, 9> U_damped = U_blocks[c];
        U_damped.diagonal().array() += lambda;

        H.block<9, 9>(start_idx, start_idx) = U_damped;
        g.segment<9>(start_idx) = g_c_blocks[c];
    }

    // 3. Заполняем диагональные блоки V и g_p
    for (size_t p = 0; p < num_points; ++p) {
        int start_idx = num_cam_params + static_cast<int>(p) * 3;
        // Добавляем демпфирование
        Eigen::Matrix<double, 3, 3> V_damped = V_blocks[p];
        V_damped.diagonal().array() += lambda;

        H.block<3, 3>(start_idx, start_idx) = V_damped;
        g.segment<3>(start_idx) = g_p_blocks[p];
    }

    // 4. Заполняем внедиагональные блоки W и W^T
    for (size_t i = 0; i < obs.size(); ++i) {
        const auto& o = obs[i];
        int cam_start = static_cast<int>(o.cam_id) * 9;
        int point_start = num_cam_params + static_cast<int>(o.point_id) * 3;

        const auto& W = W_blocks[i];

        // Верхний правый блок W (9x3)
        H.block<9, 3>(cam_start, point_start) = W;
        // Нижний левый блок W^T (3x9)
        H.block<3, 9>(point_start, cam_start) = W.transpose();
    }

    // 5. Решаем систему H * delta = -g
    Eigen::VectorXd rhs = -g;
    Eigen::VectorXd delta(total_params);
    delta.setZero();
    // Используем LDLT разложение для симметричной матрицы
    Eigen::LDLT<Eigen::MatrixXd> ldlt_solver(H);

    // Проверка на положительную определенность
    if (ldlt_solver.info() != Eigen::Success) {
        std::cerr << "[WARNING] LDLT factorization failed. Trying LLT..."
                  << std::endl;

        // Пробуем LLT
        Eigen::LLT<Eigen::MatrixXd> llt_solver(H);
        if (llt_solver.info() != Eigen::Success) {
            std::cerr << "[ERROR] LLT factorization also failed. Using "
                         "identity damping..."
                      << std::endl;

            // Добавляем сильное демпфирование и пробуем снова
            H += Eigen::MatrixXd::Identity(total_params, total_params) *
                 lambda * 10.0;
            Eigen::LLT<Eigen::MatrixXd> llt_solver2(H);
            if (llt_solver2.info() == Eigen::Success) {
                delta = llt_solver2.solve(rhs);
            } else {
                std::cerr << "[ERROR] Complete failure. Returning zero step."
                          << std::endl;
                delta = Eigen::VectorXd::Zero(total_params);
            }
        } else {
            delta = llt_solver.solve(rhs);
        }
    } else {
        delta = ldlt_solver.solve(rhs);
    }

    // 6. Разделяем решение на delta_c и delta_p
    delta_c = delta.head(num_cam_params);
    delta_p_blocks.resize(num_points);

    for (size_t p = 0; p < num_points; ++p) {
        int start_idx = num_cam_params + static_cast<int>(p) * 3;
        delta_p_blocks[p] = delta.segment<3>(start_idx);
    }

    // 7. Вычисляем нормы
    delta_c_norm = delta_c.norm();
    delta_p_norm = 0.0;
    for (const auto& dp : delta_p_blocks) {
        delta_p_norm += dp.squaredNorm();
    }
    delta_p_norm = std::sqrt(delta_p_norm);
}

auto BundleAdjuster::SolveDirect(int max_iterations, double initial_lambda)
    -> double {
    SetupProblem();
    double lambda = initial_lambda;
    double v = 2.0;
    double current_error = calculate_total_error();

    std::cout << "=================================\n"
              << "Direct (Full System) optimizer started\n"
              << "\tCameras: " << this->num_cameras
              << "\n\tPoints: " << this->num_points << "\n"
              << "Initial Error: " << current_error << std::endl;

    // Поблочные матрицы
    std::vector<Eigen::Matrix<double, 9, 9>> U_blocks(num_cameras);
    std::vector<Eigen::Matrix<double, 3, 3>> V_blocks(num_points);
    std::vector<Eigen::Vector<double, 9>> g_c_blocks(num_cameras);
    std::vector<Eigen::Vector<double, 3>> g_p_blocks(num_points);
    std::vector<Eigen::Matrix<double, 9, 3>> W_blocks(num_observations);

    auto start = std::chrono::system_clock::now();
    for (int iter = 0; iter < max_iterations; ++iter) {
        std::cout << "\n[ITERATION " << iter << "]" << std::endl;
        std::cout << "Current Error: " << current_error << std::endl;
        // 1) Построение гессиана и градиента
        BuildHessianAndGradient(U_blocks, V_blocks, g_c_blocks, g_p_blocks,
                                W_blocks);

        bool step_accepted = false;
        for (int lm_try = 0; lm_try < LM_TRIES; ++lm_try) {
            std::cout << "  [LM Trial " << lm_try << "] Lambda: " << lambda
                      << ", v: " << v << std::endl;

            Eigen::VectorXd delta_c;
            std::vector<Eigen::Vector3d> delta_p_blocks;
            double delta_c_norm = 0.0;
            double delta_p_norm = 0.0;
            // 2) Решение полной системы напрямую
            SolveFullSystemDirect(lambda, U_blocks, V_blocks, g_c_blocks,
                                  g_p_blocks, W_blocks, delta_c, delta_p_blocks,
                                  delta_c_norm, delta_p_norm);

            // Проверка корректности
            if (delta_c.size() == 0) {
                std::cout
                    << "[ERROR] Direct solver returned empty delta_c. Aborting."
                    << std::endl;
                return current_error;
            }
            std::cout << "    | Step norms: ||Delta_c|| = " << delta_c_norm
                      << ", ||Delta_p|| = " << delta_p_norm << std::endl;
            // Проверка сходимости
            if (delta_c_norm < 1e-8) {
                std::cout << "[CONVERGENCE] Delta_c norm too small. Finishing."
                          << std::endl;
                return current_error;
            }
            // 3) Пробный шаг
            std::vector<std::array<double, 9>> new_camera_params =
                camera_params;
            std::vector<std::array<double, 3>> new_point_params = point_params;
            // Применяем delta_c с учетом маски
            for (size_t c = 0; c < num_cameras; ++c) {
                for (int j = 0; j < 9; ++j) {
                    size_t idx = c * 9 + j;
                    if (idx < (size_t)delta_c.size() && camera_param_mask[j]) {
                        new_camera_params[c][j] +=
                            delta_c(static_cast<int>(idx));
                    }
                }
            }
            // Применяем delta_p
            for (size_t p = 0; p < num_points; ++p) {
                if (p < delta_p_blocks.size()) {
                    for (int j = 0; j < 3; ++j) {
                        new_point_params[p][j] += delta_p_blocks[p][j];
                    }
                }
            }
            // 4) Считаем ошибку
            double trial_error = calculate_total_error_for_params(
                new_camera_params, new_point_params);
            std::cout << "    | Trial Error: " << trial_error << std::endl;
            // 5) LM
            if (trial_error < current_error) {
                // Step ACCEPTED
                std::cout << "    | SUCCESS" << std::endl;

                camera_params = std::move(new_camera_params);
                point_params = std::move(new_point_params);
                current_error = trial_error;
                step_accepted = true;

                lambda = std::max(lambda / 3.0, 1e-7);
                v = 2.0;
                break;
            } else {
                // Step REJECTED
                std::cout << "    | FAILED" << std::endl;
                lambda *= v;
                v *= 2.0;
                step_accepted = false;
            }
        }
        if (!step_accepted) {
            std::cout
                << "[HALT] Direct method failed to find a valid step. Stopping."
                << std::endl;
            break;
        }
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "Direct method time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms\n";
    return current_error;
}