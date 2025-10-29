#ifndef _t33nsy_STRUCTS
#define _t33nsy_STRUCTS
#include <Eigen/Core>
#include <Eigen/Dense>

struct Observation {
    int cam_id;
    int point_id;
    double u, v;
};

struct Camera {
    Eigen::Vector3d rotation;  // aa
    Eigen::Vector3d translation;
    double f, k1, k2;
};

struct Point3D {
    Eigen::Vector3d pos;
};

#endif /* _t33nsy_STRUCTS */
