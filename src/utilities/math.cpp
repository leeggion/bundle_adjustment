#include "math.hpp"

// --- ИСПРАВЛЕННАЯ ФУНКЦИЯ ---
Eigen::Vector2d projection(const Camera& cam, const Point3D& p) {
    Eigen::AngleAxisd aa(cam.rotation.norm(), cam.rotation.normalized());
    Eigen::Vector3d Pc = aa.toRotationMatrix() * p.pos + cam.translation;

    // УБИРАЕМ МИНУСЫ, чтобы соответствовать BALReprojectionError
    double x = Pc[0] / Pc[2];
    double y = Pc[1] / Pc[2];
    
    double r2 = x * x + y * y;
    double radial = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2;

    double u = cam.f * radial * x;
    double v = cam.f * radial * y;

    return {u, v};
}

// Эта функция - просто тест, она не используется в main
void evaluate_autodiff(const Camera& cam, const Point3D& pt, double u,
                       double v) {
    BALReprojectionError error(u, v);
    ceres::AutoDiffCostFunction<BALReprojectionError, 2, 9, 3> cost_fn(&error);

    double cam_params[9] = {cam.rotation[0],
                            cam.rotation[1],
                            cam.rotation[2],
                            cam.translation[0],
                            cam.translation[1],
                            cam.translation[2],
                            cam.f,
                            cam.k1,
                            cam.k2};
    double pt_params[3] = {pt.pos[0], pt.pos[1], pt.pos[2]};
    double* const parameters[2] = {cam_params, pt_params};

    // buffers for residuals and jacobians
    double residuals[2];
    double* jacobians[2];
    double jac_cam[2 * 9], jac_pt[2 * 3];
    jacobians[0] = jac_cam;
    jacobians[1] = jac_pt;

    cost_fn.Evaluate(parameters, residuals, jacobians);

    Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J_cam(jac_cam);
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_point(jac_pt);

    std::cout << "Residual: "
              << Eigen::Map<Eigen::Vector2d>(residuals).transpose()
              << std::endl;
    std::cout << "Jacobian wrt camera:\n" << J_cam << std::endl;
    std::cout << "Jacobian wrt point:\n" << J_point << std::endl;
}