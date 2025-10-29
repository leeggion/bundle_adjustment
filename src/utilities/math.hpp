#ifndef _t33nsy_MATH
#define _t33nsy_MATH
#include <opencv2/core.hpp>

#include "DatasetLoader.hpp"
#include "Structs.hpp"

// shit
Eigen::Vector2d projection(const Camera& cam, const Point3D& p) {
    Eigen::AngleAxisd aa(cam.rotation.norm(), cam.rotation.normalized());
    Eigen::Vector3d Pc = aa.toRotationMatrix() * p.pos + cam.translation;

    double x = Pc[0] / Pc[2];
    double y = Pc[1] / Pc[2];
    double r2 = x * x + y * y;
    double radial = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2;

    double u = cam.f * radial * x;
    double v = cam.f * radial * y;

    return {u, v};
}

#endif /* _t33nsy_MATH */