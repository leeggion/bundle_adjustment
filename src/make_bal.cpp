#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <set>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

void visualizeMatches(const Mat& img1, const Mat& img2,
                      const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2,
                      const vector<DMatch>& matches, const string& outname) {
    Mat vis;
    hconcat(img1, img2, vis);

    for (const auto& m : matches) {
        Point2f p1 = kp1[m.queryIdx].pt;
        Point2f p2 = kp2[m.trainIdx].pt + Point2f((float)img1.cols, 0);

        line(vis, p1, p2, Scalar(255, 255, 255), 1);
    }

    imwrite(outname, vis);
    cout << "Saved visualization: " << outname << endl;
}

// Simple union-find to group matched keypoints across views into tracks
// может dsu если быстрее надо например
struct UnionFind {
    vector<int> p;
    UnionFind(int n = 0) {
        p.resize(n);
        iota(p.begin(), p.end(), 0);
    }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) p[b] = a;
    }
    int size() { return (int)p.size(); }
};

struct ImageFeatures {
    vector<KeyPoint> kps;
    Mat desc;
    Mat img;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " /path/to/images output_bal.txt\n";
        return -1;
    }
    string images_dir = argv[1];
    string out_file = argv[2];

    // 1) list images
    vector<string> img_files;
    for (auto& p : fs::directory_iterator(images_dir)) {
        if (!p.is_regular_file()) continue;
        string ext = p.path().extension().string();
        // common image extensions
        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".JPG" ||
            ext == ".PNG" || ext == ".bmp")
            img_files.push_back(p.path().string());
    }
    sort(img_files.begin(), img_files.end());
    if (img_files.size() < 2) {
        cerr << "Need at least 2 images\n";
        return -1;
    }

    // 2) SIFT detect & compute
    Ptr<SIFT> sift = SIFT::create();
    vector<ImageFeatures> feats;
    feats.resize(img_files.size());
    Size img_size;
    for (size_t i = 0; i < img_files.size(); ++i) {
        Mat I = imread(img_files[i], IMREAD_UNCHANGED);
        if (I.empty()) {
            cerr << "Failed read: " << img_files[i] << "\n";
            return -1;
        }
        if (i == 0) img_size = I.size();
        feats[i].img = I;
        sift->detectAndCompute(I, noArray(), feats[i].kps, feats[i].desc);
        cout << "Image " << i << " " << img_files[i]
             << " keypoints=" << feats[i].kps.size() << "\n";
    }

    // 3) Pairwise matching between consecutive images and build correspondences
    BFMatcher matcher(NORM_L2);
    // We'll create global indexing of keypoints: index = offset_per_image[i] +
    // kp_index_in_image
    vector<int> offset_per_image(img_files.size() + 1, 0);
    for (size_t i = 0; i < img_files.size(); ++i)
        offset_per_image[i + 1] =
            offset_per_image[i] + (int)feats[i].kps.size();
    int total_kps = offset_per_image.back();
    UnionFind uf(total_kps);

    // store matches pairs for potential triangulation: for each match we add
    // union between keypoint ids
    for (size_t i = 0; i + 1 < img_files.size(); ++i) {
        vector<vector<DMatch>> knn;
        matcher.knnMatch(feats[i].desc, feats[i + 1].desc, knn, 2);
        vector<DMatch> good;
        for (auto& v : knn) {
            if (v.size() < 2) continue;
            if (v[0].distance < 0.75f * v[1].distance) good.push_back(v[0]);
        }
        cout << "Matches between " << i << " and " << (i + 1) << " = "
             << good.size() << "\n";

        string fname =
            "./vis/vis_" + to_string(i) + "_" + to_string(i + 1) + ".png";

        visualizeMatches(feats[i].img, feats[i + 1].img, feats[i].kps,
                         feats[i + 1].kps, good, fname);
        for (auto& m : good) {
            int a = offset_per_image[i] + m.queryIdx;
            int b = offset_per_image[i + 1] + m.trainIdx;
            uf.unite(a, b);
        }
    }

    // 4) Collect tracks: map root -> vector of (image_idx, kp_idx)
    unordered_map<int, vector<pair<int, int>>> tracks_map;
    for (size_t img_i = 0; img_i < img_files.size(); ++img_i) {
        for (size_t k = 0; k < feats[img_i].kps.size(); ++k) {
            int global_idx = offset_per_image[img_i] + (int)k;
            int root = uf.find(global_idx);
            tracks_map[root].push_back({(int)img_i, (int)k});
        }
    }

    // Filter tracks: at least seen in 2 images
    vector<vector<pair<int, int>>> tracks;
    for (auto& kv : tracks_map) {
        // remove duplicates of same image (rare) - keep one per image (take
        // first)
        unordered_map<int, int> onePerImage;
        for (auto& p : kv.second) {
            if (onePerImage.find(p.first) == onePerImage.end())
                onePerImage[p.first] = p.second;
        }
        if (onePerImage.size() >= 2) {
            vector<pair<int, int>> t;
            for (auto& kv2 : onePerImage) t.push_back(kv2);
            // sort by image index
            sort(t.begin(), t.end(),
                 [](auto& a, auto& b) { return a.first < b.first; });
            tracks.push_back(t);
        }
    }
    cout << "Tracks kept (>=2 views): " << tracks.size() << "\n";

    // 5) Estimate relative poses between consecutive images and accumulate
    // camera poses We'll compute R_i, t_i such that P_cam = R_i * X_world + t_i
    // (world = cam0 frame)
    vector<Mat> Rs(img_files.size()), Ts(img_files.size());
    Rs[0] = Mat::eye(3, 3, CV_64F);
    Ts[0] = Mat::zeros(3, 1, CV_64F);

    for (size_t i = 0; i + 1 < img_files.size(); ++i) {
        // match again but keep good keypoints and use RANSAC E
        vector<vector<DMatch>> knn;
        matcher.knnMatch(feats[i].desc, feats[i + 1].desc, knn, 2);
        vector<DMatch> good;
        for (auto& v : knn) {
            if (v.size() < 2) continue;
            if (v[0].distance < 0.75f * v[1].distance) good.push_back(v[0]);
        }
        if (good.size() < 8) {
            cout << "Insufficient matches between " << i << " and " << i + 1
                 << "\n";
            Rs[i + 1] = Rs[i].clone();
            Ts[i + 1] = Ts[i].clone();
            continue;
        }

        vector<Point2f> pts1, pts2;
        for (auto& m : good) {
            pts1.push_back(feats[i].kps[m.queryIdx].pt);
            pts2.push_back(feats[i + 1].kps[m.trainIdx].pt);
        }

        // estimate focal from image size (initial guess)
        double focal = 0.9 * max(img_size.width, img_size.height);
        Point2d pp(img_size.width / 2.0, img_size.height / 2.0);

        Mat E = findEssentialMat(pts1, pts2, focal, pp, RANSAC, 0.999, 1.0);
        Mat R, t;
        Mat mask;
        int inliers = recoverPose(E, pts1, pts2, R, t, focal, pp, mask);
        cout << "recoverPose inliers=" << inliers << "\n";

        // Accumulate relative pose: pose_{i+1} = [R * pose_i.R, R * pose_i.t +
        // t]?? careful with frames If we treat X_world = X_cam0, and
        // recoverPose gives transform from cam i -> cam i+1:
        //     For a point in world (cam0) we would chain transforms. To keep it
        //     simple, we map cam i+1 pose relative to cam0:
        // using previous absolute transform T_i = [R_i, t_i] which maps
        // world->cam_i: P_i = R_i * X_world + t_i relative transform from cam_i
        // to cam_{i+1} found by recoverPose maps X_cam_i -> X_cam_{i+1}:
        // P_{i+1} = R_rel * P_i + t_rel So P_{i+1} = R_rel*(R_i*X + t_i) +
        // t_rel = (R_rel*R_i)*X + (R_rel*t_i + t_rel)
        Mat R_rel = R;
        Mat t_rel = t;
        Rs[i + 1] = R_rel * Rs[i];
        Ts[i + 1] = R_rel * Ts[i] + t_rel;
    }

    // 6) Triangulate points: for each track, choose first two views where it's
    // observed and triangulate
    struct Obs {
        int cam;
        int kp;
    };
    vector<Point3d> points3d;
    vector<vector<pair<int, Point2d>>>
        observations_for_point;  // for each 3D point: list of (cam_idx,
                                 // image_xy)

    // Precompute projection matrices for triangulation: P = [R | t] but
    // triangulate requires 3x4 matrices mapping world->camera? We'll
    // triangulate using projection from camera0 world coords. We'll compute
    // camera_i projection into camera0 coordinate system's image plane: need
    // camera matrix K * [R_i | t_i], where R_i,t_i map world->cam_i (we are
    // using world = cam0 coords).
    double focal_px = 0.9 * max(img_size.width, img_size.height);
    Mat K = (Mat_<double>(3, 3) << focal_px, 0, img_size.width / 2.0, 0,
             focal_px, img_size.height / 2.0, 0, 0, 1);

    vector<Mat> Pmats(img_files.size());
    for (size_t i = 0; i < img_files.size(); ++i) {
        Mat Rt(3, 4, CV_64F);
        Rs[i].convertTo(Rs[i], CV_64F);
        Ts[i].convertTo(Ts[i], CV_64F);
        Rs[i].copyTo(Rt(Rect(0, 0, 3, 3)));
        Ts[i].copyTo(Rt.col(3));
        Pmats[i] = K * Rt;  // 3x4
    }

    for (auto& track : tracks) {
        // pick first two views
        int camA = track[0].first, kpA = track[0].second;
        int camB = track[1].first, kpB = track[1].second;
        Point2f pA = feats[camA].kps[kpA].pt;
        Point2f pB = feats[camB].kps[kpB].pt;

        // build homogeneous points
        Mat pts4;
        Mat ptsA(2, 1, CV_64F), ptsB(2, 1, CV_64F);
        ptsA.at<double>(0, 0) = pA.x;
        ptsA.at<double>(1, 0) = pA.y;
        ptsB.at<double>(0, 0) = pB.x;
        ptsB.at<double>(1, 0) = pB.y;

        // Triangulate (needs 2xN)
        Mat ptsA2(2, 1, CV_64F), ptsB2(2, 1, CV_64F);
        ptsA2.at<double>(0, 0) = pA.x;
        ptsA2.at<double>(1, 0) = pA.y;
        ptsB2.at<double>(0, 0) = pB.x;
        ptsB2.at<double>(1, 0) = pB.y;
        Mat X;
        triangulatePoints(Pmats[camA], Pmats[camB], ptsA2, ptsB2, X);  // 4xN
        if (X.cols < 1) continue;
        Mat x4 = X.col(0);
        double w = x4.at<double>(3, 0);
        if (fabs(w) < 1e-8) continue;
        Point3d P3(x4.at<double>(0, 0) / w, x4.at<double>(1, 0) / w,
                   x4.at<double>(2, 0) / w);
        // simple cheirality check: z should be positive in both cams
        Mat Pw = (Mat_<double>(3, 1) << P3.x, P3.y, P3.z);
        Mat zA = Rs[camA] * Pw + Ts[camA];
        Mat zB = Rs[camB] * Pw + Ts[camB];
        if (zA.at<double>(2, 0) <= 0 || zB.at<double>(2, 0) <= 0) continue;

        // store point & all observations for this track (we'll include all
        // images in track for obs)
        int pt_idx = (int)points3d.size();
        points3d.push_back(P3);
        observations_for_point.emplace_back();
        for (auto& obs : track) {
            int cam = obs.first, kp = obs.second;
            Point2f p = feats[cam].kps[kp].pt;
            // convert to BAL image coords: origin center, y upward
            double x_bal = p.x - img_size.width / 2.0;
            double y_bal = -(p.y - img_size.height / 2.0);
            observations_for_point.back().push_back(
                {cam, Point2d(x_bal, y_bal)});
        }
    }
    cout << "Triangulated points: " << points3d.size() << "\n";

    // 7) Build observations list in BAL format: each obs is (cam_idx,
    // point_idx, x, y)
    vector<tuple<int, int, double, double>> observations;
    for (size_t pid = 0; pid < observations_for_point.size(); ++pid) {
        for (auto& obs : observations_for_point[pid]) {
            observations.emplace_back(obs.first, (int)pid, obs.second.x,
                                      obs.second.y);
        }
    }

    // 8) Write BAL file
    ofstream ofs(out_file);
    if (!ofs) {
        cerr << "Cannot open output " << out_file << "\n";
        return -1;
    }
    int num_cams = (int)img_files.size();
    int num_points = (int)points3d.size();
    int num_obs = (int)observations.size();
    ofs << num_cams << " " << num_points << " " << num_obs << "\n";

    // observations
    for (auto& o : observations) {
        int c, p;
        double x, y;
        tie(c, p, x, y) = o;
        ofs << c << " " << p << " " << x << " " << y << "\n";
    }

    // cameras: for each camera we must output 9 params: angle-axis (3), t (3),
    // f, k1, k2
    for (size_t i = 0; i < img_files.size(); ++i) {
        // rotation: Rs[i] is 3x3 rotation (world->cam). Convert to Rodrigues
        // (angle-axis)
        Mat rvec;
        Rodrigues(Rs[i], rvec);  // rotation vector (angle-axis)
        Vec3d tvec(Ts[i].at<double>(0, 0), Ts[i].at<double>(1, 0),
                   Ts[i].at<double>(2, 0));
        double f_init = focal_px;
        double k1 = 0.0, k2 = 0.0;
        ofs << rvec.at<double>(0, 0) << " " << rvec.at<double>(1, 0) << " "
            << rvec.at<double>(2, 0) << " " << tvec[0] << " " << tvec[1] << " "
            << tvec[2] << " " << f_init << " " << k1 << " " << k2 << "\n";
    }

    // points (3 coords each)
    for (auto& P : points3d) {
        ofs << P.x << " " << P.y << " " << P.z << "\n";
    }

    ofs.close();
    cout << "Wrote BAL file: " << out_file << "\n";
    return 0;
}
