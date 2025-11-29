import cv2
import numpy as np
import os
import sys
from PIL import Image, ExifTags

# --- Union-Find для связывания цепочек (Img0->Img1->Img2...) ---


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root1] = root2


class CameraPose:
    def __init__(self, R, t):
        self.R = R
        self.t = t
        self.r_vec, _ = cv2.Rodrigues(self.R)


class Track:
    def __init__(self, track_id):
        self.id = track_id
        self.coords = None
        # views: {camera_index: (feature_index, x, y)}
        self.views = {}
        self.export_id = -1


class SequentialSfM:
    def __init__(self, image_dir='../test6'):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(image_dir) else []
        self.images = []
        self.keypoints = []
        self.descriptors = []
        self.Ks = {}
        self.cameras = {}
        self.tracks = {}

    def get_focal_and_k(self, path, width, height):
        focal_px = None
        try:
            img = Image.open(path)
            exif = img._getexif()
            if exif:
                tags = {ExifTags.TAGS[k]: v for k,
                        v in exif.items() if k in ExifTags.TAGS}
                if 'FocalLengthIn35mmFilm' in tags:
                    f_35 = tags['FocalLengthIn35mmFilm']
                    focal_px = (f_35 * max(width, height)) / 36.0
        except:
            pass

        if focal_px is None:
            focal_px = 1.2 * max(width, height)

        K = np.array([[focal_px, 0, width / 2.0],
                      [0, focal_px, height / 2.0],
                      [0, 0, 1]], dtype=np.float64)
        return K

    def load_data(self):
        print(f"Загрузка {len(self.image_files)} изображений (Sequential)...")
        sift = cv2.SIFT_create()
        for i, fname in enumerate(self.image_files):
            path = os.path.join(self.image_dir, fname)
            img = cv2.imread(path)
            h, w = img.shape[:2]
            self.Ks[i] = self.get_focal_and_k(path, w, h)
            self.images.append(img)
            kp, des = sift.detectAndCompute(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            self.keypoints.append(kp)
            self.descriptors.append(des)

    def get_normalized_points(self, pts, K):
        pts_norm = cv2.undistortPoints(np.expand_dims(pts, axis=1), K, None)
        return pts_norm.squeeze()

    def match_sequential_and_build_tracks(self):
        print("Последовательный матчинг (0-1, 1-2, 2-3...)...")
        num_imgs = len(self.images)
        if num_imgs < 2:
            return

        uf = UnionFind()
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50))

        # Только последовательные пары
        for i in range(num_imgs - 1):
            j = i + 1
            print(f"  Matching {i} -> {j}")

            if self.descriptors[i] is None or self.descriptors[j] is None:
                continue

            matches = flann.knnMatch(
                self.descriptors[i], self.descriptors[j], k=2)
            pts1, pts2 = [], []
            indices_ij = []

            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    pts1.append(self.keypoints[i][m.queryIdx].pt)
                    pts2.append(self.keypoints[j][m.trainIdx].pt)
                    indices_ij.append((m.queryIdx, m.trainIdx))

            if len(pts1) < 15:
                continue

            # Фильтрация RANSAC для чистоты треков
            pts1_norm = self.get_normalized_points(np.array(pts1), self.Ks[i])
            pts2_norm = self.get_normalized_points(np.array(pts2), self.Ks[j])

            E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0, 0),
                                           method=cv2.RANSAC, prob=0.999, threshold=0.003)

            if mask is not None:
                mask = mask.ravel()
                for k, is_inlier in enumerate(mask):
                    if is_inlier:
                        # Объединяем feature K на кадре I с feature L на кадре J
                        uf.union((i, indices_ij[k][0]), (j, indices_ij[k][1]))

        print("Сборка треков...")
        grouped = {}
        # Проходим по всем фичам всех изображений
        for i in range(num_imgs):
            for k_idx in range(len(self.keypoints[i])):
                key = (i, k_idx)
                if key in uf.parent:
                    root = uf.find(key)
                    if root not in grouped:
                        grouped[root] = []
                    grouped[root].append(key)

        track_cnt = 0
        for features in grouped.values():
            # Удаляем дубликаты (если вдруг на одной картинке 2 фичи попали в 1 трек)
            seen_cams = set()
            clean_feats = []
            for (c, f) in features:
                if c not in seen_cams:
                    seen_cams.add(c)
                    clean_feats.append((c, f))

            if len(clean_feats) < 2:
                continue

            t = Track(track_cnt)
            for (c, f) in clean_feats:
                t.views[c] = (f, self.keypoints[c][f].pt[0],
                              self.keypoints[c][f].pt[1])
            self.tracks[track_cnt] = t
            track_cnt += 1
        print(f"Всего треков: {len(self.tracks)}")

    def triangulate_track(self, track):
        """Триангулирует трек, используя первые две доступные зарегистрированные камеры"""
        cams = sorted([c for c in track.views if c in self.cameras])
        if len(cams) < 2:
            return False

        # Обычно берем последние добавленные или самые широкие (здесь берем просто пару)
        c1, c2 = cams[0], cams[1]

        K1, p1 = self.Ks[c1], self.cameras[c1]
        K2, p2 = self.Ks[c2], self.cameras[c2]

        P1 = K1 @ np.hstack((p1.R, p1.t))
        P2 = K2 @ np.hstack((p2.R, p2.t))

        u1 = np.array(track.views[c1][1:3]).reshape(2, 1)
        u2 = np.array(track.views[c2][1:3]).reshape(2, 1)

        pt_4d = cv2.triangulatePoints(P1, P2, u1, u2)
        # Нормализация
        w = pt_4d[3]
        if abs(w) < 1e-6:
            return False
        track.coords = (pt_4d[:3] / w).ravel()
        return True

    def reconstruction_loop(self):
        if len(self.images) < 2:
            return
        print("--- Реконструкция (Sequential) ---")

        # 1. Инициализация парой 0-1
        idx1, idx2 = 0, 1
        print(f"Инициализация пары {idx1}-{idx2}")

        # Ищем общие треки
        common = [t for t in self.tracks.values(
        ) if idx1 in t.views and idx2 in t.views]
        if len(common) < 8:
            print("Слишком мало точек для инициализации.")
            return

        pts1 = np.array([t.views[idx1][1:3] for t in common])
        pts2 = np.array([t.views[idx2][1:3] for t in common])

        # Получаем позу 0->1
        E, _ = cv2.findEssentialMat(self.get_normalized_points(pts1, self.Ks[idx1]),
                                    self.get_normalized_points(
                                        pts2, self.Ks[idx2]),
                                    focal=1.0, pp=(0, 0), method=cv2.RANSAC)
        _, R, t, _ = cv2.recoverPose(E, self.get_normalized_points(pts1, self.Ks[idx1]),
                                     self.get_normalized_points(
                                         pts2, self.Ks[idx2]),
                                     focal=1.0, pp=(0, 0))

        self.cameras[idx1] = CameraPose(
            np.eye(3), np.zeros((3, 1)))  # Cam 0 в (0,0,0)
        self.cameras[idx2] = CameraPose(R, t)

        # Триангулируем начальные точки
        count_init = 0
        for t in common:
            if self.triangulate_track(t):
                count_init += 1
        print(f"Инициализировано {count_init} точек.")

        # 2. Инкрементальное добавление 2, 3, ... N
        for i in range(2, len(self.images)):
            print(f"Подключение камеры {i}...")

            # Ищем 3D точки (которые уже вычислены на этапах 0..i-1) и видны на i
            obj_pts = []
            img_pts = []

            for track in self.tracks.values():
                if track.coords is not None and i in track.views:
                    obj_pts.append(track.coords)
                    img_pts.append(track.views[i][1:3])

            if len(obj_pts) < 6:
                print(f"Камера {i} потеряла трекинг (мало точек).")
                break

            # PnP
            success, rvec, tvec, _ = cv2.solvePnPRansac(
                np.array(obj_pts), np.array(img_pts),
                self.Ks[i], None, reprojectionError=6.0, confidence=0.99
            )

            if success:
                R_new, _ = cv2.Rodrigues(rvec)
                self.cameras[i] = CameraPose(R_new, tvec)

                # Триангулируем НОВЫЕ точки, которые появились между (i-1) и i
                # (или любые старые треки, которые стали видны с новой камеры и
                # теперь имеют >=2 вида, но еще не имеют координат)
                new_pts = 0
                for track in self.tracks.values():
                    if track.coords is None and i in track.views:
                        # Проверяем, видит ли трек какая-либо из предыдущих камер
                        prev_cams = [
                            c for c in track.views if c in self.cameras and c != i]
                        if prev_cams:
                            if self.triangulate_track(track):
                                new_pts += 1
                print(f"  + Камера {i} добавлена. Новых точек: {new_pts}")
            else:
                print(f"  PnP failed for camera {i}")
                break

    def export_custom_txt(self, filename):
        """Экспорт в запрошенном формате: одна строка на параметры камеры, одна строка на координаты точки"""
        print(f"Экспорт в файл: {filename}")

        valid_cams = sorted(self.cameras.keys())
        # Маппинг индексов камер для файла (0, 1, 2...)
        cam_map = {old: new for new, old in enumerate(valid_cams)}

        valid_tracks = [t for t in self.tracks.values()
                        if t.coords is not None]
        for i, t in enumerate(valid_tracks):
            t.export_id = i

        # Сбор наблюдений
        observations = []
        for t in valid_tracks:
            for c_idx, (_, x, y) in t.views.items():
                if c_idx in cam_map:
                    K = self.Ks[c_idx]
                    # Центрируем как обычно в BAL
                    cx, cy = K[0, 2], K[1, 2]
                    observations.append(
                        (cam_map[c_idx], t.export_id, x - cx, -(y - cy)))

        with open(filename, 'w') as f:
            # HEADER
            f.write(
                f"{len(valid_cams)} {len(valid_tracks)} {len(observations)}\n")

            # OBSERVATIONS
            for o in observations:
                f.write(f"{o[0]} {o[1]} {o[2]:.4f} {o[3]:.4f}\n")

            # CAMERAS (9 параметров в одну строку)
            # R(3) t(3) f k1 k2
            for c_idx in valid_cams:
                pose = self.cameras[c_idx]
                rv = pose.r_vec.ravel()
                tv = pose.t.ravel()
                f_val = self.Ks[c_idx][0, 0]

                # Формируем строку
                # rx ry rz tx ty tz f k1 k2
                line = f"{rv[0]:.9f} {rv[1]:.9f} {rv[2]:.9f} " \
                    f"{tv[0]:.9f} {tv[1]:.9f} {tv[2]:.9f} " \
                    f"{f_val:.9f} 0.000000 0.000000\n"
                f.write(line)

            # POINTS (3 координаты в одну строку)
            # X Y Z
            for t in valid_tracks:
                f.write(
                    f"{t.coords[0]:.9f} {t.coords[1]:.9f} {t.coords[2]:.9f}\n")

        print("Готово.")


if __name__ == "__main__":
    # Папка с изображениями передается первым аргументом
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "../test6"

    sfm = SequentialSfM(input_dir)
    sfm.load_data()
    sfm.match_sequential_and_build_tracks()
    sfm.reconstruction_loop()
    sfm.export_custom_txt("../data/test6.txt")
