#include "Visualize.hpp"

void visualize_reprojection(const std::vector<Camera>& cams,
                            const std::vector<Point3D>& points,
                            const std::vector<Observation>& obs, int cam_id) {
    cv::Mat img(800, 800, CV_8UC3, cv::Scalar(20, 20, 20));

    // Сначала найдём диапазон проекций, чтобы нормализовать
    double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for (const auto& o : obs) {
        if (o.cam_id != cam_id) continue;
        Eigen::Vector2d proj = projection(cams[o.cam_id], points[o.point_id]);
        min_x = std::min(min_x, proj[0]);
        max_x = std::max(max_x, proj[0]);
        min_y = std::min(min_y, proj[1]);
        max_y = std::max(max_y, proj[1]);
    }

    double scale_x = 0.9 * img.cols / (max_x - min_x + 1e-8);
    double scale_y = 0.9 * img.rows / (max_y - min_y + 1e-8);

    // Рисуем точки
    for (const auto& o : obs) {
        if (o.cam_id != cam_id) continue;

        Eigen::Vector2d proj = projection(cams[o.cam_id], points[o.point_id]);
        int x = static_cast<int>((proj[0] - min_x) * scale_x + 0.05 * img.cols);
        int y = static_cast<int>((proj[1] - min_y) * scale_y + 0.05 * img.rows);

        // Проверка границ
        x = std::min(std::max(x, 0), img.cols - 1);
        y = std::min(std::max(y, 0), img.rows - 1);

        cv::circle(img, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow("Reprojection", img);
    cv::waitKey(0);
}

void visualize_optimized_reprojection(
    const std::vector<std::array<double, 9>>& optimized_camera_params,
    const std::vector<std::array<double, 3>>& optimized_point_params,
    const std::vector<Observation>& obs, int cam_id) {
    // 1. Преобразование optimized_camera_params в std::vector<Camera>
    std::vector<Camera> cams_struct(optimized_camera_params.size());
    for (size_t i = 0; i < optimized_camera_params.size(); ++i) {
        // Заполняем структуру Camera (или CamStruct) из массива double
        cams_struct[i].rotation =
            Eigen::Vector3d(optimized_camera_params[i].data());
        cams_struct[i].translation =
            Eigen::Vector3d(optimized_camera_params[i].data() + 3);
        cams_struct[i].f = optimized_camera_params[i][6];
        cams_struct[i].k1 = optimized_camera_params[i][7];
        cams_struct[i].k2 = optimized_camera_params[i][8];
    }
    // 2. Преобразование optimized_point_params в std::vector<Point3D>
    std::vector<Point3D> points_struct(optimized_point_params.size());
    for (size_t i = 0; i < optimized_point_params.size(); ++i) {
        // Заполняем структуру Point3D (или PointStruct) из массива double
        points_struct[i].pos =
            Eigen::Vector3d(optimized_point_params[i].data());
    }
    // 3. Вызов оригинального визуализатора
    visualize_reprojection(cams_struct, points_struct, obs, cam_id);
}