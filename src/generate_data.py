import numpy as np
import os

def rodrigues_to_matrix(r_vec):
    theta = np.linalg.norm(r_vec)
    if theta < 1e-6: return np.eye(3)
    k = r_vec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

def look_at(eye, target, up=np.array([0, 1, 0])):
    """Создает матрицу вращения (R), чтобы камера смотрела из eye на target."""
    z_axis = eye - target       
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.array([x_axis, y_axis, z_axis])
    return R

def matrix_to_rodrigues(R):
    """Преобразует матрицу вращения обратно в вектор Родригеса."""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(theta))
    return axis * theta

def project_point(point_3d, camera_params):
    r_vec = np.array(camera_params[0:3])
    t_vec = np.array(camera_params[3:6])
    f, k1, k2 = camera_params[6], camera_params[7], camera_params[8]

    R = rodrigues_to_matrix(r_vec)
    p_cam = np.dot(R, point_3d) + t_vec

    if p_cam[2] >= 0: return None 

    p_norm = -p_cam[0:2] / p_cam[2]
    r_sq = np.dot(p_norm, p_norm)
    distortion = 1.0 + k1 * r_sq + k2 * (r_sq**2)
    return p_norm * distortion * f

def generate_gn_friendly(num_cameras=3, num_points=10, filename="gn_dataset.txt", noise_scale=0.01):
    np.random.seed(42)
    
    print(f"--- GN Mode: Noise Scale = {noise_scale} ---")

    points_gt = (np.random.rand(num_points, 3) - 0.5) * 4.0 

    cameras_gt = []
    center_point = np.array([0.0, 0.0, 0.0])

    for i in range(num_cameras):
        angle = (i - num_cameras/2) * (np.pi / 8) 
        radius = 15.0
        
        pos = np.array([np.sin(angle)*radius, 0, -np.cos(angle)*radius])
        R_mat = look_at(pos, center_point) 
        
        # В модель BAL мы подаем t = -R * C (где C - центр проекции в мире)
        # Но в формуле проекции p = R*X + t. Значит t_vec - это трансляция.
        # Внимание: В формуле P_cam = R * P_world + t
        # Если камера стоит в 'pos', то P_cam должн быть 0, если P_world == pos.
        # 0 = R * pos + t  =>  t = -R * pos
        
        t_vec = -np.dot(R_mat, pos)
        r_vec = matrix_to_rodrigues(R_mat)
        
        f = 1000.0
        k1, k2 = 0.0, 0.0 
        cameras_gt.append(np.concatenate([r_vec, t_vec, [f, k1, k2]]))

    observations = []
    point_visibility = np.zeros(num_points) 

    for cam_idx, cam_params in enumerate(cameras_gt):
        for pt_idx, pt_3d in enumerate(points_gt):
            proj = project_point(pt_3d, cam_params)
            if proj is not None and abs(proj[0]) < 2000:
                observations.append((cam_idx, pt_idx, proj[0], proj[1]))
                point_visibility[pt_idx] += 1

    valid_pt_indices = np.where(point_visibility >= 2)[0]
    final_observations = [obs for obs in observations if obs[1] in valid_pt_indices]

    old_to_new_idx = {old: new for new, old in enumerate(valid_pt_indices)}

    remapped_observations = []
    for cam_idx, old_pt_idx, x, y in final_observations:
        remapped_observations.append((cam_idx, old_to_new_idx[old_pt_idx], x, y))
    
    points_gt_clean = points_gt[valid_pt_indices]
    
    print(f"Изначально точек: {num_points}, Оставлено (2+ view): {len(points_gt_clean)}")

    cameras_out = [c.copy() for c in cameras_gt]
    points_out = [p.copy() for p in points_gt_clean]

    for c in cameras_out:
        c[0:3] += np.random.randn(3) * (0.1 * noise_scale) 
        c[3:6] += np.random.randn(3) * (1.0 * noise_scale)  
    
    for p in points_out:
        p += np.random.randn(3) * (1.0 * noise_scale)

    # 6. Запись (Flat Format)
    with open(filename, 'w') as f:
        f.write(f"{num_cameras} {len(points_gt_clean)} {len(remapped_observations)}\n")
        
        for obs in remapped_observations:
            f.write(f"{obs[0]} {obs[1]} {obs[2]:.6f} {obs[3]:.6f}\n")
            
        for cam in cameras_out:
            line = " ".join(f"{val:.8f}" for val in cam)
            f.write(line + "\n")
                
        for pt in points_out:
            line = " ".join(f"{val:.8f}" for val in pt)
            f.write(line + "\n")

    print(f"Файл {filename} создан.")

# Уровень 1: Микро-шум. Проверка, что алгоритм вообще работает (делает шаги в верном направлении).
generate_gn_friendly(3, 10, "../data/gn_easy.txt", noise_scale=0.001)

# Уровень 2: Умеренный шум. Реальный тест.
generate_gn_friendly(3, 10, "../data/gn_medium.txt", noise_scale=0.05)

# Уровень 3: Граница стабильности GN.
generate_gn_friendly(5, 20, "../data/gn_hard.txt", noise_scale=0.2)