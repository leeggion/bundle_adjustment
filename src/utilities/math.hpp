#ifndef _t33nsy_MATH
#define _t33nsy_MATH
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/core.hpp>

#include "DatasetLoader.hpp"
#include "Structs.hpp"

// shit
Eigen::Vector2d projection(const Camera& cam, const Point3D& p);

struct BALReprojectionError {
    BALReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point,
                    T* residuals) const {
        // camera = [angle_axis(3), translation(3), f, k1, k2]
        Eigen::Matrix<T, 3, 1> p(point[0], point[1], point[2]);

        Eigen::Matrix<T, 3, 1> aa(camera[0], camera[1], camera[2]);
        Eigen::Matrix<T, 3, 1> Pc;
        ceres::AngleAxisRotatePoint(aa.data(), p.data(), Pc.data());

        Pc += Eigen::Matrix<T, 3, 1>(camera[3], camera[4], camera[5]);

        T x = Pc[0] / Pc[2];
        T y = Pc[1] / Pc[2];
        T r2 = x * x + y * y;
        T radial = T(1.0) + camera[7] * r2 + camera[8] * r2 * r2;
        T u = camera[6] * radial * x;
        T v = camera[6] * radial * y;

        residuals[0] = u - T(observed_x);
        residuals[1] = v - T(observed_y);
        return true;
    }

    static ceres::CostFunction* Create(double observed_x, double observed_y) {
        return new ceres::AutoDiffCostFunction<BALReprojectionError, 2, 9, 3>(
            new BALReprojectionError(observed_x, observed_y));
    }

    double observed_x, observed_y;
};

#endif /* _t33nsy_MATH */
