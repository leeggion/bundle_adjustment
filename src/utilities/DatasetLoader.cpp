#include "DatasetLoader.hpp"

DatasetLoader::DatasetLoader(std::string path) noexcept { this->path = path; }

auto DatasetLoader::load() noexcept -> bool {
    try {
        this->file = std::ifstream(this->path);
    } catch (std::exception e) {
        std::cout << e.what() << std::endl;
        return false;
    }
    int n_cams, n_points, n_obs;
    file >> n_cams >> n_points >> n_obs;
    (this->cams).resize(n_cams);
    (this->points).resize(n_points);
    (this->obs).resize(n_obs);
    for (auto& o : obs) file >> o.cam_id >> o.point_id >> o.u >> o.v;
    for (auto& c : cams) {
        file >> c.rotation[0] >> c.rotation[1] >> c.rotation[2];
        file >> c.translation[0] >> c.translation[1] >> c.translation[2];
        file >> c.f >> c.k1 >> c.k2;
    }
    for (auto& p : points) file >> p.pos[0] >> p.pos[1] >> p.pos[2];
    return true;
}