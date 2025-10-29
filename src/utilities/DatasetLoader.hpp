#ifndef _t33nsy_DATASETLOADER
#define _t33nsy_DATASETLOADER

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Structs.hpp"

class DatasetLoader {
   public:
    DatasetLoader(std::string) noexcept;
    ~DatasetLoader() = default;
    auto load() noexcept -> bool;
    auto get_path() noexcept -> std::string { return path; }
    auto get_cams() noexcept -> std::vector<Camera> { return cams; }
    auto get_points() noexcept -> std::vector<Point3D> { return points; }
    auto get_observations() noexcept -> std::vector<Observation> { return obs; }

   private:
    std::string path;
    std::ifstream file;
    std::vector<Camera> cams;
    std::vector<Point3D> points;
    std::vector<Observation> obs;
};

#endif /* _t33nsy_DATASETLOADER */
