#ifndef _t33nsy_VISUALIZE
#define _t33nsy_VISUALIZE
#include <opencv2/opencv.hpp>

#include "Structs.hpp"
#include "math.hpp"
void visualize_reprojection(const std::vector<Camera>& cams,
                            const std::vector<Point3D>& points,
                            const std::vector<Observation>& obs, int cam_id);

void visualize_optimized_reprojection(
    const std::vector<std::array<double, 9>>& optimized_camera_params,
    const std::vector<std::array<double, 3>>& optimized_point_params,
    const std::vector<Observation>& obs, int cam_id);
#endif /* _t33nsy_VISUALIZE */