#ifndef _t33nsy_MATH
#define _t33nsy_MATH
#include <opencv2/core.hpp>

#include "DatasetLoader.hpp"
#include "Structs.hpp"

// shit
Eigen::Vector2d projection(const Camera& cam, const Point3D& p);

#endif /* _t33nsy_MATH */
