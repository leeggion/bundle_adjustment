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
