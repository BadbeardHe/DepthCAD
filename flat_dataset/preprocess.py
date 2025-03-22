import os
import numpy as np
import cv2


def sqrt_ldr(correlations):
    tof_conf = np.abs(correlations[0, :, :]) + np.abs(correlations[1, :, :])
    tof_conf_l = 16 * np.sqrt(tof_conf + 36) - 96
    tof_conf[tof_conf == 0] = 1
    i_tmp = tof_conf_l * correlations[0, :, :] / tof_conf
    q_tmp = tof_conf_l * correlations[1, :, :] / tof_conf

    return np.stack((i_tmp, q_tmp), axis=0)


def load_raw(scene, sqrt_in=True):
    shape = [424, 512, 9]
    ori_correlations = np.fromfile(scene, dtype=np.float32).reshape(shape)

    target_shape = [512, 512, 9]
    correlations = np.zeros(target_shape, dtype=np.float32)
    correlations = np.nan_to_num(correlations, nan=0, neginf=0, posinf=0)

    for i in range(9):
        correlations[:, :, i] = cv2.resize(ori_correlations[:, :, i], (512, 512), interpolation=cv2.INTER_LINEAR)

    tof_IQ_40 = np.stack((correlations[:, :, 1], correlations[:, :, 0]), axis=0)
    tof_IQ_30 = np.stack((correlations[:, :, 4], correlations[:, :, 3]), axis=0)
    tof_IQ_58 = np.stack((correlations[:, :, 7], correlations[:, :, 6]), axis=0)

    if sqrt_in:
        tof_IQ_40 = sqrt_ldr(tof_IQ_40)
        tof_IQ_30 = sqrt_ldr(tof_IQ_30)
        tof_IQ_58 = sqrt_ldr(tof_IQ_58)

    tof_IQs = np.stack((
        tof_IQ_30[0], tof_IQ_30[1],
        tof_IQ_40[0], tof_IQ_40[1],
        tof_IQ_58[0], tof_IQ_58[1]
    ), axis=0)

    tof_IQs = np.nan_to_num(tof_IQs, nan=0, neginf=0, posinf=0)
    return tof_IQs


def compute_gradient_confidence(depth_map):
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    confidence_map = 1 - grad_magnitude
    return confidence_map


if __name__ == '__main__':
    root = '/Path/to/FLAT/Ideal/Dir'
    idxs = os.listdir(root)
    idxs.sort()

    ideal_root = '/Path/to/FLAT/Ideal/Dir'
    ideal_norm_root = 'data/ideal_IQ'

    noise_root = '/Path/to/FLAT/Noise/Dir'
    noise_norm_root = 'data/noise_IQ'

    noise_depth_root = '/Path/to/FLAT/Noise/Depth/Dir'
    conf_root = 'data/confidence'
    
    roots_to_check = [ideal_norm_root, noise_norm_root, conf_root]
    for root in roots_to_check:
        if not os.path.exists(root):
            os.mkdir(root)

    suffixes = ['A', 'B', 'C', 'D', 'E', 'F']
    for idx in idxs:
        ideal_path = os.path.join(ideal_root, idx)
        ideal_IQs = load_raw(ideal_path)

        noise_path = os.path.join(noise_root, idx)
        noise_IQs = load_raw(noise_path)
        
        ideal_IQs /= max(noise_IQs.max(), abs(noise_IQs.min()))    
        noise_IQs /= max(noise_IQs.max(), abs(noise_IQs.min()))    
        
        noise_depth = np.load(f"{noise_depth_root}/{idx}.npy")
        noise_depth = cv2.resize(noise_depth, (512, 512))
        confidence = compute_gradient_confidence(noise_depth)
        conf_path = f"{conf_root}/{idx}.npy"
        np.save(conf_path, confidence)

        for i in range(6):
            ideal_norm_path = f"{ideal_norm_root}/{idx}_{suffixes[i]}.npy"
            noise_norm_path = f"{noise_norm_root}/{idx}_{suffixes[i]}.npy"
            np.save(ideal_norm_path, ideal_IQs[i])
            np.save(noise_norm_path, noise_IQs[i])
